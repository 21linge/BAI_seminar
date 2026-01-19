import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import imageio
import mediapy
import re

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CTM_ROOT = os.path.join(BASE_DIR, "continuous_thought_machines")

if CTM_ROOT not in sys.path:
    sys.path.insert(0, CTM_ROOT)

#from continuous_thought_machines.models.ctm import ContinuousThoughtMachine as CTM
from ctm_variants.ctm_relu import ContinuousThoughtMachineReLU as CTM
from continuous_thought_machines.data.custom_datasets import MazeImageFolder
from continuous_thought_machines.tasks.mazes.plotting import make_maze_gif
from continuous_thought_machines.tasks.image_classification.plotting import plot_neural_dynamics

def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

def maze_loss(predictions, certainties, targets, cirriculum_lookahead=5, use_most_certain=True):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, route_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, route_length]

    cirriculum_lookahead: how far to look ahead in the auto-cirriculum

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    
    """
    # Predictions reshaped to: [B*route_length, 5, internal_ticks]
    predictions_reshaped = predictions.flatten(0,1)
    # Targets reshaped to: [B*route_length, internal_ticks]
    targets_reshaped = torch.repeat_interleave(targets.unsqueeze(-1), 
                                               predictions.size(-1), -1).flatten(0,1).long()
    
    # Losses are of shape [B, route_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions_reshaped, targets_reshaped)
    losses = losses.reshape(predictions[:,:,0].shape)
    
    # Below is the code for auto-cirriculum
    # Find where correct, and make sure to always push +5 beyond that
    iscorrects = (predictions.argmax(2) == targets.unsqueeze(-1)).cumsum(1)
    correct_mask = (iscorrects == torch.arange(1, iscorrects.size(1)+1, device=iscorrects.device).reshape(1, -1, 1))
    correct_mask[:,0,:] = 1
    upto_where = correct_mask.cumsum(1).argmax(1).max(-1)[0]+cirriculum_lookahead
    loss_mask = torch.zeros_like(losses)
    for bi in range(predictions.size(0)):
        loss_mask[bi, :upto_where[bi]] = 1

    # Reduce losses along route dimension
    # Will now be of shape [B, internal_ticks]
    losses = (losses * loss_mask).sum(1)/(loss_mask.sum(1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1]
    loss_selected = losses[batch_indexer, loss_index_2]

    loss = ((loss_minimum_ce + loss_selected)/2).mean()
    return loss, loss_index_2, upto_where.detach().cpu().numpy()



def make_pbar_desc(train_loss, train_accuracy_step, train_accuracy_maze, test_loss, test_accuracy_step, test_accuracy_maze, lr, where_most_certain, upto_where):
    """A helper function to create a description for the tqdm progress bar"""
    pbar_desc = (
        f"Train Loss={train_loss:0.3f}. "
        f"Train Step Acc={train_accuracy_step:0.3f}. "
        f"Train Maze Acc={train_accuracy_maze:0.3f}. "
        f"Test Loss={test_loss:0.3f}. "
        f"Test Step Acc={test_accuracy_step:0.3f}. "
        f"Test Maze Acc={test_accuracy_maze:0.3f}. "
        f"LR={lr:0.6f}. "
    )
    pbar_desc += (
        f"Where_certain={where_most_certain.float().mean().item():0.2f}"
        f"+-{where_most_certain.float().std().item():0.2f} "
        f"({where_most_certain.min().item():d}<->{where_most_certain.max().item():d}). "
        f"Upto={sum(upto_where) / len(upto_where):0.2f}."
    )
    return pbar_desc


def update_training_curve_plot(fig, ax1, ax2, train_losses, test_losses, train_accuracies_step, train_accuracies_maze, test_accuracies_step, test_accuracies_maze, steps):
    clear_output(wait=True)
    
    # Plot loss
    ax1.clear()
    ax1.plot(range(len(train_losses)), train_losses, 'b-', alpha=0.7, label=f'Train Loss: {train_losses[-1]:.3f}')
    ax1.plot(steps, test_losses, 'r-', marker='o', label=f'Test Loss: {test_losses[-1]:.3f}')
    ax1.set_title('Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy (step and maze)
    ax2.clear()
    ax2.plot(range(len(train_accuracies_step)), train_accuracies_step, 'b-', alpha=0.7, label=f'Train Step Acc: {train_accuracies_step[-1]:.3f}')
    ax2.plot(range(len(train_accuracies_maze)), train_accuracies_maze, 'g--', alpha=0.7, label=f'Train Maze Acc: {train_accuracies_maze[-1]:.3f}')
    ax2.plot(steps, test_accuracies_step, 'r-', marker='o', label=f'Test Step Acc: {test_accuracies_step[-1]:.3f}')
    ax2.plot(steps, test_accuracies_maze, color='orange', linestyle='--', marker='o', label=f'Test Maze Acc: {test_accuracies_maze[-1]:.3f}')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    display(fig)

def train(
    model,
    trainloader,
    testloader,
    device='cpu',
    training_iterations=10000,
    test_every=1000,            # kept but no longer used (epoch-based validation)
    checkpoint_every=10000,
    lr=1e-4,
    log_dir='./logs'
):

    def get_latest_checkpoint(log_dir):
        files = [f for f in os.listdir(log_dir) if re.match(r'checkpoint_\d+\.pt', f)]
        return (
            os.path.join(
                log_dir,
                max(files, key=lambda f: int(re.search(r'\d+', f).group()))
            )
            if files else None
        )

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if latest_checkpoint_path := get_latest_checkpoint(log_dir):
        checkpoint = torch.load(latest_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        train_accuracies_step = checkpoint['train_accuracies_step']
        train_accuracies_maze = checkpoint['train_accuracies_maze']
        test_accuracies_step = checkpoint['test_accuracies_step']
        test_accuracies_maze = checkpoint['test_accuracies_maze']
        steps = checkpoint['steps']
        start_iter = checkpoint['step']
    else:
        train_losses = []
        test_losses = []
        train_accuracies_step = []
        train_accuracies_maze = []
        test_accuracies_step = []
        test_accuracies_maze = []
        steps = []
        start_iter = 0

    iterator = iter(trainloader)
    steps_per_epoch = len(trainloader)


    with tqdm(total=training_iterations, initial=start_iter) as pbar:
        for stepi in range(start_iter, training_iterations):

            # --------------------------------------------------
            # Epoch bookkeeping
            # --------------------------------------------------
            epoch = stepi // steps_per_epoch
            is_new_epoch = (stepi % steps_per_epoch == 0)

            if is_new_epoch:
                print(f"\n===== Epoch {epoch} =====")

            # --------------------------------------------------
            # Get batch
            # --------------------------------------------------
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions_raw, certainties, _ = model(inputs)
            predictions = predictions_raw.reshape(
                predictions_raw.size(0), -1, 5, predictions_raw.size(-1)
            )

            train_loss, where_most_certain, upto_where = maze_loss(
                predictions, certainties, targets, use_most_certain=True
            )

            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

            train_predictions_most_certain = predictions.argmax(2)[
                torch.arange(predictions.size(0), device=predictions.device),
                :,
                where_most_certain
            ]

            train_accuracy_step = (
                train_predictions_most_certain == targets
            ).float().mean().item()

            train_accuracy_maze = (
                train_predictions_most_certain == targets
            ).all(-1).float().mean().item()

            train_accuracies_step.append(train_accuracy_step)
            train_accuracies_maze.append(train_accuracy_maze)

            # --------------------------------------------------
            # Step-level TensorBoard logging
            # --------------------------------------------------
            writer.add_scalar("loss/train", train_losses[-1], stepi)
            writer.add_scalar("acc/train_step", train_accuracy_step, stepi)
            writer.add_scalar("acc/train_maze", train_accuracy_maze, stepi)

            # --------------------------------------------------
            # Validation (epoch-based)
            # --------------------------------------------------
            if is_new_epoch:
                model.eval()
                with torch.no_grad():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    for inputs, targets in testloader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        predictions_raw, certainties, _ = model(inputs)
                        predictions = predictions_raw.reshape(
                            predictions_raw.size(0), -1, 5, predictions_raw.size(-1)
                        )

                        test_loss, where_most_certain_test, _ = maze_loss(
                            predictions, certainties, targets, use_most_certain=True
                        )

                        all_test_losses.append(test_loss.item())
                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain_test)

                    all_test_predictions = torch.cat(all_test_predictions, dim=0)
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                    test_loss = sum(all_test_losses) / len(all_test_losses)
                    test_losses.append(test_loss)

                    all_test_predictions_most_certain = all_test_predictions.argmax(2)[
                        torch.arange(all_test_predictions.size(0), device=predictions.device),
                        :,
                        all_test_where_most_certain
                    ]

                    test_accuracy_step = (
                        all_test_predictions_most_certain == all_test_targets
                    ).float().mean().item()

                    test_accuracy_maze = (
                        all_test_predictions_most_certain == all_test_targets
                    ).all(-1).float().mean().item()

                    test_accuracies_step.append(test_accuracy_step)
                    test_accuracies_maze.append(test_accuracy_maze)

                    writer.add_scalar("loss/test", test_loss, stepi)
                    writer.add_scalar("acc/test_step", test_accuracy_step, stepi)
                    writer.add_scalar("acc/test_maze", test_accuracy_maze, stepi)

                    steps.append(stepi)

                model.train()

            # --------------------------------------------------
            # Epoch-level TensorBoard logging
            # --------------------------------------------------
            if is_new_epoch and stepi > 0:
                writer.add_scalar(
                    "epoch/train_loss",
                    np.mean(train_losses[-steps_per_epoch:]),
                    epoch
                )
                writer.add_scalar(
                    "epoch/train_step_acc",
                    np.mean(train_accuracies_step[-steps_per_epoch:]),
                    epoch
                )
                writer.add_scalar(
                    "epoch/train_maze_acc",
                    np.mean(train_accuracies_maze[-steps_per_epoch:]),
                    epoch
                )

            # --------------------------------------------------
            # Checkpointing
            # --------------------------------------------------
            if stepi % checkpoint_every == 0 or stepi == training_iterations - 1:
                checkpoint_path = os.path.join(log_dir, f'checkpoint_{stepi}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': stepi,
                    'steps': steps,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies_step': train_accuracies_step,
                    'train_accuracies_maze': train_accuracies_maze,
                    'test_accuracies_step': test_accuracies_step,
                    'test_accuracies_maze': test_accuracies_maze,
                }, checkpoint_path)
            
            pbar_desc = make_pbar_desc(
                train_loss=train_losses[-1],
                train_accuracy_step=train_accuracies_step[-1],
                train_accuracy_maze=train_accuracies_maze[-1],
                test_loss=test_losses[-1],
                test_accuracy_step=test_accuracies_step[-1],
                test_accuracy_maze=test_accuracies_maze[-1],
                lr=optimizer.param_groups[-1]["lr"],
                where_most_certain=where_most_certain,
                upto_where=upto_where
            )
            if is_new_epoch:
                pbar.set_description(pbar_desc)
                pbar.update(1)

    writer.close()
    return model

def create_maze_gif_visualization(model, testloader, device, log_dir):
    model.eval()
    with torch.no_grad():
        inputs_viz, targets_viz = next(iter(testloader))
        inputs_viz = inputs_viz.to(device)
        targets_viz = targets_viz.to(device)
        
        batch_index_to_viz = 0
        
        predictions_raw, certainties, _, pre_activations, post_activations, attention_tracking = model(inputs_viz, track=True)
        
        # Reshape predictions
        predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))
        
        # Reshape attention tracking for visualization
        att_shape = (model.kv_features.shape[2], model.kv_features.shape[3])
        attention_tracking = attention_tracking.reshape(attention_tracking.shape[0], attention_tracking.shape[1], -1, att_shape[0], att_shape[1])

        plot_neural_dynamics(post_activations, 100, log_dir, axis_snap=True)
        
        # Create maze GIF with attention visualization
        maze_input = (inputs_viz[batch_index_to_viz].detach().cpu().numpy() + 1) / 2
        maze_predictions = predictions[batch_index_to_viz].detach().cpu().numpy()
        maze_targets = targets_viz[batch_index_to_viz].detach().cpu().numpy()
        maze_attention = attention_tracking[:, batch_index_to_viz] if attention_tracking.ndim > 2 else attention_tracking

        # Generate the maze GIF
        make_maze_gif(
            maze_input,
            maze_predictions,
            maze_targets,
            maze_attention,
            log_dir
        )
        
        predictions_raw, certainties, _ = model(inputs_viz)
        predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))

def main():    
    set_seed(42)

    data_root = './small-mazes'
    train_dir = f"{data_root}/train/0"
    images = os.listdir(train_dir)[:4]

    train_data = MazeImageFolder(root=f'{data_root}/train/', which_set='train', maze_route_length=50)
    test_data = MazeImageFolder(root=f'{data_root}/test/', which_set='test', maze_route_length=50)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Training setup")
    print(f"Device           : {device}")
    print(f"PyTorch version  : {torch.__version__}")

    if device == "cuda":
        print(f"CUDA available   : {torch.cuda.is_available()}")
        print(f"CUDA version     : {torch.version.cuda}")
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA             : not available (CPU)")
    print("=" * 60)

    # Define the model
    model = CTM(
        iterations=50,
        d_model=1024,
        d_input=256,
        heads=8,
        n_synch_out=256,
        n_synch_action=256,
        synapse_depth=8,
        memory_length=15,
        deep_nlms=True,
        memory_hidden_dims=16,
        backbone_type='resnet34-2',
        out_dims=50 * 5,
        prediction_reshaper=[50, 5],
        dropout=0.1,
        do_layernorm_nlm=False,
        positional_embedding_type='none',
        neuron_select_type='random-pairing',  
    ).to(device)

    # Initialize model parameters with dummy forward pass
    print("About to run dummy forward...")
    sample_batch = next(iter(trainloader))
    dummy_input = sample_batch[0][:1].to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    model = train(model=model, trainloader=trainloader, testloader=testloader, device=device, training_iterations=100001, test_every=1000, checkpoint_every=10000, lr=1e-4, log_dir='./maze_logs')

if __name__ == "__main__":
    main()
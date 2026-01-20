# BAI_seminar

# install pip packages
pip install -r requirements.txt

# Download additional content
# run following in comand line
git clone https://github.com/SakanaAI/continuous-thought-machines.git
gdown "https://drive.google.com/uc?id=1Z8FFnZ7pZcu7DfoSyfy-ghWa08lgYl1V"

# Win:
Expand-Archive -Path small-mazes.zip -DestinationPath small-mazes
# (get the small mazes folder out of the small mazes folder)


# Linux:
unzip "small-mazes.zip"

# Rename "continuous-though-machines" to "continuous_though_machines"

### tensorboard:
tensorboard --logdir ./maze_logs

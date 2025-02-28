# Run Instructions

## Setup

If you are using ubuntu or macos, you should simply be able to install the python packages in a virtual environment and go. This has been tested on python3.10 but should work for newer

```
$ python -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

## Running

Once everything is installed the script can be run like so

```
python src/analyse_video.py --input-file path/to/video.mp4
```

When the application is running, press the 'q' key to quit. There are two additional optional args that can be supplied to the script.

- `--output-log`: can be used to write the event log to a file as well as printing to the terminal
- `--no-display`: option to run "headless" and not play back the video with detections overlayed

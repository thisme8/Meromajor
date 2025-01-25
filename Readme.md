# type these one by one to mitigate error in modules

step 1 => PYTHONPATH=/path/to/project python /path/to/project/modules/dense_motion.py(or any file inside module)

step 2 =>
/path to project/python -modules.dense_motion
"/path-to-project/env/bin/python -m modules.dense_motion(or any file name inside module)"

# save the two vox.tar files in a folder named extract

# when the demo asks for a config prompt paste:

python3 demo.py --config ./config/vox-256.yaml \ --checkpoint ./extract/vox-cpk.pth.tar \ --source_image ./Inputs/Monalisa.png \ --driving_video ./Inputs/driving_video.mp4 \ --result_video ./output/output.mp4 \ --relative \ --adapt_scale

(flash) administrator@administrator-MS-7E32:~/flash/UI-tars$ python uitars_worker.py --host 0.0.0.0 --port 40000 --worker http://localhost:40000 --model-path ByteDance-Seed/UI-TARS-2B-SFT --no-register --device cuda --use-flash-attn




(flash) administrator@administrator-MS-7E32:~/flash/UI-tars/ScreenAgent/client$ python run_controller.py -c config.yml

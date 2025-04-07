
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --batch=38 --limit=0 || true
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --batch=40 --limit=0 || true
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --batch=38 --kfold=10 --limit=0 || true
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --batch=40 --kfold=10 --limit=0 || true


CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --maxiter=3000 --batch=38 --limit=0 || true
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --maxiter=3000 --batch=40 --kfold=10 --limit=0 || true
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --maxiter=3000 --batch=38 --kfold=10 --limit=0 || true
CUDA_VISIBLE_DEVICES=2 /anaconda3/envs/brain/bin/python /brainnet/eegall/main.py --loss='sce' --graph='harm' --maxiter=3000 --batch=40 --kfold=10 --limit=0 || true


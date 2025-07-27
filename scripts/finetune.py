import subprocess
import time
import sys
import argparse 
from vemol.chem_utils.dataset_path import get_task_type
from pathlib import Path 
from vemol.utils.utils import get_result_file

def main(args: argparse.Namespace):
    seed = args.seed
    num_layers = args.num_layers
    model_path = args.model_path
    op = args.op
    lr = args.lr
    max_trials = args.max_trials
    scaffold_id_start = args.scaffold_id_start
    # 初始化时间计数器
    start_time = time.time()

    # 定义数据集和 scaffold 的数组
    datasets = ['polymer_xc', 'polymer_egc', 'polymer_egb', 'polymer_eea', 'polymer_ei',
             'polymer_eps', 'polymer_nc', 'polymer_eat']
    datasets = [f"{dataset}_aug" for dataset in datasets]
    
    scaffolds = [scaffold_id_start+i for i in range(5)]
    total_exps = len(datasets)*len(scaffolds)
    # scaffolds = [0, 1]
    d_model = args.d_model
    model_name = model_path.split('/')[-1]
    
    results_dir_name = f"results_finetune_polymer_aug_mc_{op}_gt{num_layers}_dmodel{d_model}_lr{lr}/{model_name}"
    # print(results_dir_name);exit()
    results_dir = Path(results_dir_name)
    dataset_cls = 'mol_graph_mpp'  # 关键变量
    model = 'base_graph_transformer'
    ema_decay = 0.5
    # 计算初始剩余任务数
    remaining_tasks = 0 
    for dataset in datasets:
        for scaffold in scaffolds:
            file = get_result_file(results_dir, 
                                       f"{dataset_cls}_{dataset}",
                                       scaffold,  
                                       model, 
                                       d_model, 
                                       num_layers,
                                       ema_decay, 
                                       seed)
            # print(file, file.exists());exit()
            if not file.exists():
                remaining_tasks += 1
    print(f"Remaining tasks: {remaining_tasks}/{total_exps}")
    # 遍历数据集和 scaffold 的组合
    for _ in range(max_trials):
        run_exps = 0
        for dataset in datasets:
            task_type = get_task_type(dataset)
            for scaffold in scaffolds:
                file = get_result_file(results_dir, 
                                       f"{dataset_cls}_{dataset}",
                                       scaffold,  
                                       model, 
                                       d_model, 
                                       num_layers,
                                       ema_decay, 
                                       seed)
                if file.exists():
                    # print(f"Skip {file}")
                    continue
                print(f"dataset={dataset} scaffold={scaffold} seed={seed} num_layers={num_layers}")
                
                command = [
                    "vemol-train",
                    "--config-name", "mpp_gt",
                    f"common.epochs=50",
                    f"dataset.scaffold_id={scaffold}",
                    f"dataset.name={dataset}",
                    f"common.seed={seed}",
                    f"common.save_results=True",
                    f'common.save_results_dir={results_dir_name}',
                    f"checkpoint.save_to_disk=False",
                    f"checkpoint.load_model={model_path}",
                    f"checkpoint.load_no_strict=True",
                    "dataset/mol_graph_form_cfg=polymer_atom_graph",
                    f"dataset.mol_graph_form_cfg.op={op}",
                    f"dataset.n_jobs=8", # 和num_workers有点混淆
                    f"model.num_layers={num_layers}",
                    f"model.d_model={d_model}",
                    f"common.ema_decay=0.5",
                    f"scheduler.lr={lr}",
                    "dataset.mol_graph_form_cfg.main_chain_embed=True",
                ]
                if task_type == 'reg':
                    command.append(f"metric=graph_reg")
                    command.append(f"criterion=mse")

                # 运行命令并捕获输出
                try:
                    result = subprocess.run(
                        command,
                        check=True,
                        text=True,
                        # stdout=subprocess.PIPE,
                        # stderr=subprocess.PIPE
                    )
                    # print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while executing command: {e.cmd}")
                    print(f"Exit code: {e.returncode}")
                    print(f"Error output: {e.stderr}")
        if run_exps == 0:
            print("finish all exps")
            break 
        else:
            print(f"finish {run_exps/total_exps} exps")
    # 计算并打印耗时
    elapsed_time = int(time.time() - start_time)
    print(f"耗时: {elapsed_time} 秒")

if __name__ == "__main__":
    # 从命令行获取种子（seed）参数
    # seed_input = sys.argv[1]
    parser = argparse.ArgumentParser("Test EMA")
    parser.add_argument("--seed", type=int, help="Random seed", required=True)
    parser.add_argument("--num_layers", type=int, default=6, help="num of layers")
    parser.add_argument("--d_model", type=int, default=512, help="d_model")
    parser.add_argument("--model_path", type=str, help="model path of pre-trained model", required=True)
    parser.add_argument("--op", type=str, help="op of star", required=True)
    parser.add_argument("--lr", type=str, help="lr of finetune", required=True)
    parser.add_argument("--max_trials", type=int, default=5, help="max trials of the exp")
    parser.add_argument("--scaffold_id_start", type=int, default=0, help="start of scaffold_id")
    args = parser.parse_args()
    main(args)
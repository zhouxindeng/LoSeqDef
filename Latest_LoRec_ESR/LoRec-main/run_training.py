from meta_config import args
from utls.trainer import SASrecTrainer
from utls.utilize import init_run

def main():
    # 配置参数
    args.model = "SASrec"
    args.dataset = "MIND"
    args.LLM = "Llama2_13"
    args.use_LLM = True
    args.device_id = "0"
    args.device = "gpu"

    # 初始化日志路径和随机种子
    log_path = f"./log/{args.model}/{args.LLM}/{args.dataset}/"
    init_run(log_path=log_path, args=args, seed=2023)

    # 加载并初始化训练器
    print("Initializing trainer...")
    trainer = SASrecTrainer(vars(args))  # 将 args 转换为字典传入
    print("Trainer initialized.")

    # 开始训练
    print("Starting training...")
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    main()

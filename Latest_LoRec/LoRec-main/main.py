from meta_config import args
from utls.model_config import *
#from utls.trainer import *
from utls.AlphaRec import *
from utls.utilize import init_run, restore_stdout_stderr

def main(seed=2023, main_file=""):
    args.seed = seed
    #args.dataset="Arts"

    # Initialize the log path & seedty
    if args.use_LLM:
        path = f"/Users/changliu/Documents/GraduationProject/Latest_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/{main_file}/" if main_file != "" else f"/root/tf-logs/lc_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/"
        if args.with_lct:
            path = f"/Users/changliu/Documents/GraduationProject/Latest_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/LCT/{main_file}/" if main_file != "" else f"/root/tf-logs/lc_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/FD/"
    else:
        path = f"/Users/changliu/Documents/GraduationProject/Latest_LoRec/LoRec-main/log/{args.model}/No_LLM/{args.dataset}/{main_file}/"  if main_file != "" else f"/root/tf-logs/lc_LoRec/LoRec-main/log/{args.model}/No_LLM/{args.dataset}/"
    init_run(log_path=path, args=args, seed=args.seed)

    glo = globals()
    global_config = vars(args)
    global_config["main_file"] = main_file

    # LCT config
    global_config["lct_config"] = glo["get_LCT_config"](global_config)
    
    # Backbone Model config
    global_config["model_config"] = glo[f"get_{global_config['model']}_config"](global_config)
    global_config['checkpoints'] = 'checkpoints'

    # Initialize correspondding trainer 
    #trainer =  glo[f"{global_config['model']}Trainer"](global_config)
    #trainer.train()

    trainer =  glo[f"AlphaRecTrainer"](global_config)
    #trainer.train()
    trainer._load_model("/Users/changliu/Documents/GraduationProject/checkpoints/AlphaRec/Arts/20250606/attack_bandwagon_1_202506062047.pth")
    trainer.visualize_feature_spaces(connection_indices=[5040, 59130, 6516, 60039, 1842, 33985, 37846, 1753, 57739, 47416],save_visualization="feature_spaces2.png",load_tsne="T-SNE.json")
    restore_stdout_stderr()


if __name__ == '__main__':    
    main_file = datetime.now().strftime('%Y%m%d')
    main(seed=2023, main_file=main_file)


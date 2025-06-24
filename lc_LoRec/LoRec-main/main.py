from meta_config import args
from utls.model_config import *
from utls.trainer import *
from utls.utilize import init_run, restore_stdout_stderr
from LLMFineTuner.llm_finetuner import *
from LLMFineTuner.llm_finetunerCPT import *

def main(seed=2023, main_file=""):
    args.seed = seed
    args.dataset="Test"

    # Initialize the log path & seedty
    if args.use_LLM:
        path = f"/Users/changliu/Documents/GraduationProject/lc_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/{main_file}/" if main_file != "" else f"/root/tf-logs/lc_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/"
        if args.with_lct:
            path = f"/Users/changliu/Documents/GraduationProject/lc_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/LCT/{main_file}/" if main_file != "" else f"/root/tf-logs/lc_LoRec/LoRec-main/log/{args.model}/{args.LLM}/{args.dataset}/FD/"
    else:
        path = f"/Users/changliu/Documents/GraduationProject/lc_LoRec/LoRec-main/log/{args.model}/No_LLM/{args.dataset}/{main_file}/"  if main_file != "" else f"/root/tf-logs/lc_LoRec/LoRec-main/log/{args.model}/No_LLM/{args.dataset}/"
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
    #Llama2FineTunerCPT(global_config)
    trainer =  glo[f"{global_config['model']}Trainer"](global_config)
    trainer.train()
    
    restore_stdout_stderr()


if __name__ == '__main__':    
    main_file = datetime.now().strftime('%Y%m%d')
    main(seed=2023, main_file=main_file)


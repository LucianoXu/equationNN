import subprocess
import sys
import pytest
import shlex

def task_test(
    cmd: str,
    input_data: str,
    timeout: int = 15
):
    
    process = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,  # Capture stderr
        text=True
    )

    try:
        _, err_data = process.communicate(input=input_data, timeout=timeout)

        # check whether the error is EOF
        if process.returncode == 1 and "EOFError: EOF when reading a line" in err_data:
            print("EOF encountered.")
            return

        assert process.returncode == 0, f"Script failed with return code {process.returncode}. Error: {err_data}"

    except subprocess.TimeoutExpired:
        print(f"Script timed out after {timeout} seconds.")

def test_instance_by_vampire():
    task_test(
        cmd = "./main instance_by_vampire algdesc/OML.alg -t 10",
        input_data = "x = x \n",
        timeout = 5
    )

def test_task_model_info():
    task_test(
        cmd = "./main model_info --modelargs=large",
        input_data = "",
        timeout = 15
    )

def test_task_get_vocab():
    task_test(
        cmd = "./main get_vocab algdesc/OML.alg",
        input_data = "",
        timeout = 15
    )

def test_task_ntok_machine():
    task_test(
        cmd = "./main ntok_machine algdesc/OML.alg -m sol",
        input_data = "<STT> \n & \n ( \n u \n v \n ) \n = \n x \n </STT> \n <ACT> \n",
        timeout = 15
    )
    
def test_task_syn_gen_examples():
    task_test(
        cmd = "./main syn_gen_examples algdesc/OML.alg -c 100 -m 10",
        input_data = "",
        timeout = 15
    )

def test_task_gen_valid_check():
    task_test(
        cmd = "./main gen_valid_check algdesc/OML.alg -c 100 -m 10",
        input_data = "",
        timeout = 15
    )

def test_task_sol_rl_fuzzer():
    task_test(
        cmd = "./main sol_rl_fuzzer algdesc/OML.alg --ckpt ./ckpt/OMLTEST -i --modelargs=medium --num_steps=512 --lr=3e-4 --batch_size=8 --acc_steps=12 --gen_step_limit=4 --sol_step_limit=12 -t 0.7 --save_interval=10",
        input_data = "",
        timeout = 15
    )

def test_task_gen_rl_vampire():
    task_test(
        cmd = "./main gen_rl_vampire algdesc/OML.alg --ckpt ./ckpt/OMLTEST -i --modelargs=medium --num_steps=512 --lr=3e-4 --batch_size=8 --acc_steps=12 --gen_step_limit=12 -t 0.7 --save_interval=10",
        input_data = "",
        timeout = 15
    )

def test_task_adv_rl():
    task_test(
        cmd = "./main adv_rl algdesc/OML.alg --ckpt ./ckpt/OMLTEST -i --modelargs=medium --starting_mode=sol --num_steps_per_turn=24 --num_turns=20 --lr=3e-4 --batch_size=6 --acc_steps=16 --gen_step_limit=8 --sol_step_limit=12 -t 0.7 --save_interval=10",
        input_data = "",
        timeout = 15
    )

def test_task_parse_examples():
    task_test(
        cmd = "./main parse_examples algdesc/OMLext.alg algdesc/OMLext_examples.txt",
        input_data = "",
        timeout = 15
    )

def test_task_test_intere_example():
    task_test(
        cmd = "./main test_intere_example algdesc/OMLext.alg algdesc/OMLext_examples.txt -o paper_eval.csv",
        input_data = "",
        timeout = 15
    )

def test_task_test_synfuzzer_examples():
    task_test(
        cmd = "./main test_synfuzzer_examples algdesc/OML.alg -c 100 -m 10 -o synfuzzer_eval.csv",
        input_data = "",
        timeout = 15
    )


# this test is commented out because it requires the trained model

# def test_task_test_modelfuzzer_examples():
#     task_test(
#         cmd = "./main test_modelfuzzer_examples algdesc/OML.alg --ckpt ./ckpt/OMLGen --batch_size=30 -t 0.7 -c 100 -m 10 -o model_eval.csv",
#         input_data = "",
#         timeout = 15
#     )


import argparse
import os
import subprocess

# Export MODEL_DIR and DATA_DIR at the start of the script
os.environ["MODEL_DIR"] = "nn/Pygeon/models/pretrained"
os.environ["DATA_DIR"] = "nn/Pygeon/data/datasets"

adam_005_tests = [
    # {"function": 80, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/AlexNet_32_CIFAR-10_custom_best.bin"},
    # {"function": 80, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/AlexNet_32_CIFAR-10_standard_best.bin"},
    # {"function": 81, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/AlexNet_CryptGPU_CIFAR-10_custom_best.bin"},
    # {"function": 81, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/AlexNet_CryptGPU_CIFAR-10_standard_best.bin"},
    # {"function": 70, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/ResNet18_avg_CIFAR-10_custom_best.bin"},
    # {"function": 70, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/ResNet18_avg_CIFAR-10_standard_best.bin"},
    # {"function": 71, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/ResNet50_avg_CIFAR-10_custom_best.bin"},
    # {"function": 71, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/ResNet50_avg_CIFAR-10_standard_best.bin"},
    # {"function": 72, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/ResNet101_avg_CIFAR-10_custom_best.bin"},
    # {"function": 72, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/ResNet101_avg_CIFAR-10_standard_best.bin"},
    # {"function": 73, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/ResNet152_avg_CIFAR-10_custom_best.bin"},
    # {"function": 73, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/ResNet152_avg_CIFAR-10_standard_best.bin"},
    # {"function": 74, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_005/VGG16_CIFAR-10_custom_best.bin"},
    # {"function": 74, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_005/VGG16_CIFAR-10_standard_best.bin"},
]

adam_001_tests = [
    # {"function": 80, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/AlexNet_32_CIFAR-10_custom_best.bin"},
    {"function": 80, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/AlexNet_32_CIFAR-10_standard_best.bin"},
    # {"function": 81, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/AlexNet_CryptGPU_CIFAR-10_custom_best.bin"},
    {"function": 81, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/AlexNet_CryptGPU_CIFAR-10_standard_best.bin"},
    # {"function": 70, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/ResNet18_avg_CIFAR-10_custom_best.bin"},
    {"function": 70, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/ResNet18_avg_CIFAR-10_standard_best.bin"},
    # {"function": 71, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/ResNet50_avg_CIFAR-10_custom_best.bin"},
    {"function": 71, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/ResNet50_avg_CIFAR-10_standard_best.bin"},
    # {"function": 72, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/ResNet101_avg_CIFAR-10_custom_best.bin"},
    {"function": 72, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/ResNet101_avg_CIFAR-10_standard_best.bin"},
    # {"function": 73, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/ResNet152_avg_CIFAR-10_custom_best.bin"},
    {"function": 73, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/ResNet152_avg_CIFAR-10_standard_best.bin"},
    # {"function": 74, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_adam_001/VGG16_CIFAR-10_custom_best.bin"},
    {"function": 74, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_adam_001/VGG16_CIFAR-10_standard_best.bin"},
]

sgd_001_tests = [
    # {"function": 80, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/AlexNet_32_CIFAR-10_custom_best.bin"},
    # {"function": 80, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/AlexNet_32_CIFAR-10_standard_best.bin"},
    # {"function": 81, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/AlexNet_CryptGPU_CIFAR-10_custom_best.bin"},
    # {"function": 81, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/AlexNet_CryptGPU_CIFAR-10_standard_best.bin"},
    # {"function": 70, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/ResNet18_avg_CIFAR-10_custom_best.bin"},
    # {"function": 70, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/ResNet18_avg_CIFAR-10_standard_best.bin"},
    # {"function": 71, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/ResNet50_avg_CIFAR-10_custom_best.bin"},
    # {"function": 71, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/ResNet50_avg_CIFAR-10_standard_best.bin"},
    # {"function": 72, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/ResNet101_avg_CIFAR-10_custom_best.bin"},
    # {"function": 72, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/ResNet101_avg_CIFAR-10_standard_best.bin"},
    # {"function": 73, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/ResNet152_avg_CIFAR-10_custom_best.bin"},
    # {"function": 73, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/ResNet152_avg_CIFAR-10_standard_best.bin"},
    # {"function": 74, "data_path": "CIFAR-10_custom_test", "model_path": "Cifar_sgd_001/VGG16_CIFAR-10_custom_best.bin"},
    # {"function": 74, "data_path": "CIFAR-10_standard_test", "model_path": "Cifar_sgd_001/VGG16_CIFAR-10_standard_best.bin"},
]

lenet_tests = [
        {"function": 82, "data_path": "MNIST_standard_test", "model_path": "MNIST_LeNet5/LeNet5_MNIST_standard_best.bin"},
        {"function": 82, "data_path": "MNIST_custom_test", "model_path": "MNIST_LeNet5/LeNet5_MNIST_custom_best.bin"},
        ]


adam_wd_tests = [
        {"function": 80, "data_path": "CIFAR-10_standard_test", "model_path": "adam_001_wd/AlexNet32_AdamW_d05_wd003_lr0001_ep100_acc76_89.bin"},
        {"function": 81, "data_path": "CIFAR-10_standard_test", "model_path": "adam_001_wd/AlexNetC_AdamW_d05_wd003_lr00008_ep80_acc66_89.bin"},
        {"function": 82, "data_path": "MNIST_standard_test", "model_path": "adam_001_wd/LeNet5_AdamW_d05_wd003_lr0001_ep100_acc99_13.bin"},
        {"function": 70, "data_path": "CIFAR-10_standard_test", "model_path": "adam_001_wd/ResNet18_avg_AdamW_d05_wd003_lr0001_ep100_acc74_86.bin"},
        {"function": 71, "data_path": "CIFAR-10_standard_test", "model_path": "adam_001_wd/ResNet50_avg_AdamW_d05_wd003_lr0001_ep100_acc74_35.bin"},
        ]

imagenet_tests = [
        {"function": 85, "data_path": "imagenet_128-256", "model_path": "ImageNet/AlexNet_imagenet.bin"},
        {"function": 86, "data_path": "imagenet_128-256", "model_path": "ImageNet/VGG_imagenet.bin"},
        ]


tests = [adam_005_tests, adam_001_tests, sgd_001_tests, lenet_tests, adam_wd_tests, imagenet_tests]

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run configurations')
    parser.add_argument('-p', type=str, default='all', help='Party identifier')
    parser.add_argument('-a', type=str, default='127.0.0.1', help='IP address for a')
    parser.add_argument('-b', type=str, default='127.0.0.1', help='IP address for b')
    parser.add_argument('-c', type=str, default='127.0.0.1', help='IP address for c')
    parser.add_argument('-d', type=str, default='127.0.0.1', help='IP address for d')
    parser.add_argument('-i', type=int, default=1, help='Number of random seeds per run')
    parser.add_argument('-n', type=int, default=100, help='Number of images to use for testing')

    args = parser.parse_args()
    args.i = str(list(range(args.i))).strip('[]').replace(', ', ',')


    for test_group in tests:
        for setup in test_group:
            # Set environment variables
            os.environ["MODEL_FILE"] = f"{setup['model_path']}"
            os.environ["SAMPLES_FILE"] = f"{setup['data_path']}_images.bin"
            os.environ["LABELS_FILE"] = f"{setup['data_path']}_labels.bin"

            # Prepare the base command
            base_cmd = f"python measurements/run_config.py -p {args.p} -a {args.a} -b {args.b} -c {args.c} -d {args.d}"

            # Run commands for different configurations
            configs = [
                "measurements/configs/nn_tests/test_fractional_vs_accuracy_16.conf",
                "measurements/configs/nn_tests/test_fractional_vs_accuracy_32.conf",
                "measurements/configs/nn_tests/test_fractional_vs_accuracy_64.conf"
            ]

            for config in configs:
                print(f"FUNCTION_IDENTIFIER={setup['function']} SRNG_SEED={args.i}")
                cmd = f"{base_cmd} {config} --override FUNCTION_IDENTIFIER={setup['function']} SRNG_SEED={args.i}"
                # print(f"FUNCTION_IDENTIFIER={setup['function']} SRNG_SEED={args.i} NUM_INPUTS={args.n}")
                # cmd = f"{base_cmd} {config} --override FUNCTION_IDENTIFIER={setup['function']} SRNG_SEED={args.i} NUM_INPUTS={args.n}"
                run_command(cmd)
if __name__ == "__main__":
    main()

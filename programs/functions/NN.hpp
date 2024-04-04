#pragma once
#include "../../protocols/Protocols.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../protocols/Matrix_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../datatypes/k_clear.hpp"
/* #include "boolean_adder.hpp" */
#include "boolean_adder_bandwidth.hpp"

#include "boolean_adder_msb.hpp"
#include "ppa_msb.hpp"
#include "ppa.hpp"
#include "ppa_msb_unsafe.hpp"
#include "ppa_msb_4_way.hpp"



#include "../../utils/print.hpp"

#include <cmath>
#include <sys/types.h>



#include "headers/simple_nn.h"
#include "architectures/CNNs.hpp"
#include "architectures/ResNet.hpp"
#include "architectures/DeepReduce.hpp"
#include "headers/config.h"

#include "Relu.hpp"
#include "arg_max.hpp"
#define FUNCTION inference
#define RESULTTYPE DATATYPE
using namespace std;
using namespace simple_nn;
using namespace Eigen;


void generateElements()
{

}


    template<typename Share>
void inference(DATATYPE* res)
{
    using FLOATTYPE = float;
    using SHARETYPE = Wrapper<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL, DATATYPE>;
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;
#if BASETYPE == 0
    using modeltype = A;
#else
    using modeltype = sint;
#endif


    /* int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10; */
    /* auto model = DRD_C100_230K<modeltype>(num_classes); */

#if FUNCTION_IDENTIFIER == 70 || FUNCTION_IDENTIFIER == 170 || FUNCTION_IDENTIFIER == 270 || FUNCTION_IDENTIFIER == 370
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet18<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 71 || FUNCTION_IDENTIFIER == 171 || FUNCTION_IDENTIFIER == 271 || FUNCTION_IDENTIFIER == 371
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 72 || FUNCTION_IDENTIFIER == 172 || FUNCTION_IDENTIFIER == 272 || FUNCTION_IDENTIFIER == 372
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet101<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 73 || FUNCTION_IDENTIFIER == 173 || FUNCTION_IDENTIFIER == 273 || FUNCTION_IDENTIFIER == 373
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet152<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 74 || FUNCTION_IDENTIFIER == 174 || FUNCTION_IDENTIFIER == 274 || FUNCTION_IDENTIFIER == 374
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 75 || FUNCTION_IDENTIFIER == 175 || FUNCTION_IDENTIFIER == 275 || FUNCTION_IDENTIFIER == 375
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet18<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 76 || FUNCTION_IDENTIFIER == 176 || FUNCTION_IDENTIFIER == 276 || FUNCTION_IDENTIFIER == 376
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 77 || FUNCTION_IDENTIFIER == 177 || FUNCTION_IDENTIFIER == 277 || FUNCTION_IDENTIFIER == 377
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet101<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 78 || FUNCTION_IDENTIFIER == 178 || FUNCTION_IDENTIFIER == 278 || FUNCTION_IDENTIFIER == 378
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet152<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 79 || FUNCTION_IDENTIFIER == 179 || FUNCTION_IDENTIFIER == 279 || FUNCTION_IDENTIFIER == 379
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 80 || FUNCTION_IDENTIFIER == 180 || FUNCTION_IDENTIFIER == 280 || FUNCTION_IDENTIFIER == 380
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = AlexNet_32<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 81 || FUNCTION_IDENTIFIER == 181 || FUNCTION_IDENTIFIER == 281 || FUNCTION_IDENTIFIER == 381
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = AlexNet_CryptGpu<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 82 || FUNCTION_IDENTIFIER == 182 || FUNCTION_IDENTIFIER == 282 || FUNCTION_IDENTIFIER == 382
    int n_test = NUM_INPUTS*BASE_DIV, ch = 1, h = 28, w = 28, num_classes = 10;
    auto model = LeNet<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 83 || FUNCTION_IDENTIFIER == 183 || FUNCTION_IDENTIFIER == 283 || FUNCTION_IDENTIFIER == 383
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = AlexNet_CryptGpu<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 84 || FUNCTION_IDENTIFIER == 184 || FUNCTION_IDENTIFIER == 284 || FUNCTION_IDENTIFIER == 384
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 100;
    auto model = DRD_C100_7K<modeltype>(num_classes);
#endif

	Config cfg;
    cfg.mode = "test";
    /* cfg.save_dir = "./SimpleNN/model_zoo/pthAdamModels/cifar10_ADAM_001/"; */
    /* cfg.save_dir = "./SimpleNN/model_zoo"; */
    /* cfg.save_dir = "./SimpleNN/model_zoo/"; */
    /* cfg.data_dir = "./SimpleNN/dataset"; */
    /* cfg.pretrained = "dummy.dummy"; */
    /* cfg.image_file = "MNIST_standard_test_images.bin"; */
    /* cfg.label_file = "MNIST_standard_test_labels.bin"; */
    /* cfg.pretrained = "pretrained_models/LeNet5/LeNet5_MNIST_standard_best.bin"; */

    /* cfg.pretrained = "old/vgg16_cifar.bin"; */
    /* cfg.image_file = "old/cifar10-test-images.bin"; */
    /* cfg.label_file = "old/cifar10-test-labels.bin"; */
    /* cfg.pretrained = "VGG16_CIFAR-10_standard_best.bin"; */
    /* cfg.image_file = "CIFAR-10_standard_test_images.bin"; */
    /* cfg.label_file = "CIFAR-10_standard_test_labels.bin"; */
    /* cfg.pretrained = "VGG16_CIFAR-10_custom_best.bin"; */
    /* cfg.image_file = "CIFAR-10_custom_test_images.bin"; */
    /* cfg.label_file = "CIFAR-10_custom_test_labels.bin"; */
  

#if MODELOWNER != -1 || DATAOWNER != -1
// Open the configuration file
    FILE* file = fopen("config.txt", "r");
    if (file == NULL) {
        std::cerr << "Error opening file" << std::endl;
        exit(1);
    }

    // Buffer to hold each line from the file
    char line[256];

    // Read the file line by line
    while (fgets(line, sizeof(line), file)) {
        // Get the key part of the line
        char* key = strtok(line, "=");
        // Get the value part of the line
        char* value = strtok(NULL, "\n");

        if (key && value) {
            // Compare the key and assign the value to the appropriate field in cfg
            /* std::cout << " Key: " << key << " Value: " << value << std::endl; */
            if (strcmp(key, "mode") == 0) {
                cfg.mode = value;
            }
            else if (strcmp(key, "save_dir") == 0) {
                cfg.save_dir = value;
            }
            else if (strcmp(key, "data_dir") == 0) {
                cfg.data_dir = value;
            }
            else if (strcmp(key, "pretrained") == 0) {
                cfg.pretrained = value;
            }
            else if (strcmp(key, "image_file") == 0) {
                cfg.image_file = value;
            }
            else if (strcmp(key, "label_file") == 0) {
                cfg.label_file = value;
            }
        }
    }

    // Close the file after reading
    fclose(file);

    // Print out the loaded configuration values
    print_online("Mode: " + cfg.mode);
    print_online("Save Directory: " + cfg.save_dir);
    print_online("Data Directory: " + cfg.data_dir);
    print_online("Pretrained Model: " + cfg.pretrained);
    print_online("Image File: " + cfg.image_file);
    print_online("Label File: " + cfg.label_file);

#endif

    /* fclose(file); */
    //file should look like this:
    //mode=test
    //save_dir=./SimpleNN/model_zoo

    cfg.batch = NUM_INPUTS*(BASE_DIV);
    

    /* cfg.batch = 1*BASE_DIV; */
    /* using DATATYPE = uint32_t; */
    /* const int parallel_factor = 1; */
    /* using cleartype = k_clear<A>; */
    /* cfg.pretrained = "model_DRD_C100_230K.bin"; */
    /* cfg.pretrained = "resnet50_cifar100.bin"; */
    /* cfg.image_file = "CIFAR-100_test_images.bin"; */
    /* cfg.label_file = "CIFAR-100_test_labels.bin"; */
    /* cfg.pretrained = "alex32.bin"; */
    /* cfg.pretrained = "lenet5_avg.pth"; */
    /* cfg.pretrained = "ResNet50_Cifar_Max.bin"; */

    /* using Sharetype = Wrapper<DATATYPE>; */
    /* using F = FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE> ; */

   
    /* // Array of string literals */
    /* const char* literals[] = {"simplenn", "--mode=test", "--save_dir=./SimpleNN/model_zoo", "--pretrained=model_DRD_C100_230K.bin"}; */

    /* // Number of arguments */
    /* size_t numArgs = sizeof(literals) / sizeof(literals[0]); */

    /* // Allocate an array of char* pointers */
    /* char** argv = new char*[numArgs]; */

    /* // Copy each string literal into a new dynamically allocated string */
    /* for (size_t i = 0; i < numArgs; ++i) { */
    /*     size_t length = std::strlen(literals[i]); */
    /*     argv[i] = new char[length + 1];  // +1 for the null terminator */
    /*     std::strcpy(argv[i], literals[i]); */
    /* } */ 
    /* int argc = 4; */
	/* cfg.parse(argc, argv); */
	/* cfg.print_config(); */

#if IS_TRAINING == 1
	MatX<float> train_X;
    VecXi train_Y;
    int n_train = 60000;
#endif

#if JIT_VEC == 0
#if IS_TRAINING == 1
	DataLoader<modeltype> train_loader;
#endif
	DataLoader<modeltype> test_loader;
#else
#if IS_TRAINING == 1
	DataLoader<float> train_loader;
#endif
    DataLoader<float> test_loader;
#endif

#if IS_TRAINING == 1
		train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train);
		train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train);
        MatX<A> train_XX = train_X.unaryExpr([](float val) { 
                A tmp;
                tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val));
                return tmp;
        /* MatX<modeltype> train_XX = train_X.unaryExpr([](float val) { */ 
        /*         modeltype tmp; */
        /*         tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); */
        /*         return tmp; */
    /* return sint(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); */
});

                modeltype::communicate();
            for (int i = 0; i < train_XX.size(); i++) {
                train_XX(i).template complete_receive_from<P_0>();
            }

		/* train_loader.load(train_XX, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train); */
#endif
#if DATAOWNER == -1
    auto test_Y = read_dummy_labels(n_test);
    auto test_X = read_dummy_images(n_test, ch, h, w);
#else
#if PSELF == DATAOWNER
    print_online("Reading dataset from file...");
		/* auto test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
    string path = cfg.data_dir + "/" + cfg.image_file;
    auto test_X = read_custom_images(path, n_test, ch, h, w);
	/* auto test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
    print_online("Dataset imported.");
#else
    auto test_X = read_dummy_images(n_test, ch, h, w);
#endif
    string lpath = cfg.data_dir + "/" + cfg.label_file;
    auto test_Y = read_custom_labels(lpath, n_test);
	/* auto test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test); */
#endif
/* #endif */

#if JIT_VEC == 0 
        MatX<modeltype> test_XX = test_X.unaryExpr([](float val) { 
                modeltype tmp;
                tmp.template prepare_receive_and_replicate<DATAOWNER>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val));
                return tmp;
});

                modeltype::communicate();
            for (int i = 0; i < test_XX.size(); i++) {
                test_XX(i).template complete_receive_from<DATAOWNER>();
            }
	/* test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test); */
print_online("Received Secret Share of Dataset");
#endif



#if JIT_VEC == 0
test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);
#else
test_loader.load(test_X, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);
#endif


    /* ReducedNet<modeltype> model = DRD_C100_230K<modeltype>(num_classes); */
    /* ResNet<modeltype> model = ResNet50<modeltype>(num_classes); */
    /* AlexNet_32<modeltype> model = AlexNet_32<modeltype>(num_classes); */
    /* VGG<modeltype> model = VGG<modeltype>(num_classes); */
	/* load_model(cfg, model); */
    
    /* if (cfg.mode == "train") { */
        /* if (cfg.loss == "cross_entropy") { */
        /*     model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new CrossEntropyLoss<sint>); */
        /* } */
        /* else { */
        /*     model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new MSELoss<sint>); */
        /* } */
        /* model.fit(train_loader, cfg.epoch, test_loader); */
        /* model.save("./model_zoo", cfg.model + ".pth"); */
    /* } */
    
    /* else { */
/* std::cout << "Compiling model..." << std::endl; */
print_online("Compiling model...");
        model.compile({ cfg.batch/(BASE_DIV), ch, h, w });
/* std::cout << "Loading Model Parameters..." << std::endl; */
print_online("Loading model Parameters...");
#if PSELF == MODELOWNER
        print_online("Loading model parameters from file...");
#endif
#if MODELOWNER != -1
        model.template load<MODELOWNER>(cfg.save_dir, cfg.pretrained);
#endif
        print_online("Received Secret Share of Model Parameters.");
#if FUNCTION_IDENTIFIER < 300 // otherwise only measure secret sharing of model
        model.evaluate(test_loader);
#endif
    /* } */

	/* return 0; */
}

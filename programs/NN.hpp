#pragma once
#include "architectures/CNNs.hpp"        // includes common CNN architectures
#include "architectures/DeepReduce.hpp"  // includes DeepReduce architectures
#include "architectures/ResNet.hpp"      // includes ResNet architectures
#include "functions/GEMM.hpp"            // for Conv2d
#include "functions/Relu.hpp"            // for Activations
#include "functions/max_min.hpp"         // for MaxPool2d and Softmax
#include "functions/prob_div.hpp"        //for AvgPool2d
#include "headers/config.h"              // NN configuration
#include "headers/simple_nn.h"           // templated inference engine
#if TRUNC_APPROACH > 1
#include "functions/exact_truncation.hpp"
#endif
#define FUNCTION inference
#define RESULTTYPE DATATYPE
using namespace std;
using namespace simple_nn;
using namespace Eigen;

template <typename Share>
void inference(DATATYPE* res)
{
    using LFLOATTYPE = float;
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;

#if BASETYPE == 0
    using modeltype = A;
#else
    using modeltype = sint;
#endif

    // === Specify Model Archiecture and Data Dimensions ===

#if FUNCTION_IDENTIFIER == 70 || FUNCTION_IDENTIFIER == 170 || FUNCTION_IDENTIFIER == 270 || FUNCTION_IDENTIFIER == 370
    // ResNet18, CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet18<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 71 || FUNCTION_IDENTIFIER == 171 || FUNCTION_IDENTIFIER == 271 || \
    FUNCTION_IDENTIFIER == 371
    // ResNet50, CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 72 || FUNCTION_IDENTIFIER == 172 || FUNCTION_IDENTIFIER == 272 || \
    FUNCTION_IDENTIFIER == 372
    // ResNet101, CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet101<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 73 || FUNCTION_IDENTIFIER == 173 || FUNCTION_IDENTIFIER == 273 || \
    FUNCTION_IDENTIFIER == 373
    // ResNet152, CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet152<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 74 || FUNCTION_IDENTIFIER == 174 || FUNCTION_IDENTIFIER == 274 || \
    FUNCTION_IDENTIFIER == 374
    // VGG16, CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 75 || FUNCTION_IDENTIFIER == 175 || FUNCTION_IDENTIFIER == 275 || \
    FUNCTION_IDENTIFIER == 375
    // ResNet18, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet18<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 76 || FUNCTION_IDENTIFIER == 176 || FUNCTION_IDENTIFIER == 276 || \
    FUNCTION_IDENTIFIER == 376
    // ResNet50, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 77 || FUNCTION_IDENTIFIER == 177 || FUNCTION_IDENTIFIER == 277 || \
    FUNCTION_IDENTIFIER == 377
    // ResNet101, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet101<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 78 || FUNCTION_IDENTIFIER == 178 || FUNCTION_IDENTIFIER == 278 || \
    FUNCTION_IDENTIFIER == 378
    // ResNet152, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet152<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 79 || FUNCTION_IDENTIFIER == 179 || FUNCTION_IDENTIFIER == 279 || \
    FUNCTION_IDENTIFIER == 379
    // VGG16, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 80 || FUNCTION_IDENTIFIER == 180 || FUNCTION_IDENTIFIER == 280 || \
    FUNCTION_IDENTIFIER == 380
    // AlexNet, CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = AlexNet_32<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 81 || FUNCTION_IDENTIFIER == 181 || FUNCTION_IDENTIFIER == 281 || \
    FUNCTION_IDENTIFIER == 381
    // AlexNet (as proposed by CryptGPU), CIFAR-10
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = AlexNet_CryptGpu<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 82 || FUNCTION_IDENTIFIER == 182 || FUNCTION_IDENTIFIER == 282 || \
    FUNCTION_IDENTIFIER == 382
    // LeNet5, MNIST
    int n_test = NUM_INPUTS * BASE_DIV, ch = 1, h = 28, w = 28, num_classes = 10;
    auto model = LeNet<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 83 || FUNCTION_IDENTIFIER == 183 || FUNCTION_IDENTIFIER == 283 || \
    FUNCTION_IDENTIFIER == 383
    // AlexNet (as proposed by CryptGPU), ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = AlexNet_CryptGpu<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 84 || FUNCTION_IDENTIFIER == 184 || FUNCTION_IDENTIFIER == 284 || \
    FUNCTION_IDENTIFIER == 384
    // DeepReduce C100,7K, CIFAR-100
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 100;
    auto model = DRD_C100_7K<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 85 || FUNCTION_IDENTIFIER == 185 || FUNCTION_IDENTIFIER == 285 || \
    FUNCTION_IDENTIFIER == 385
    // AlexNet PyTorch, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = AlexNet_PyTorch<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 86 || FUNCTION_IDENTIFIER == 186 || FUNCTION_IDENTIFIER == 286 || \
    FUNCTION_IDENTIFIER == 386
    // VGG16 PyTorch, ImageNet
    int n_test = NUM_INPUTS * BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = VGG16_PyTorch<modeltype>(num_classes);
#endif

    // === Read Labels and Images ===

    Config cfg;
    cfg.mode = "test";  // Training is not supported yet
    cfg.save_dir = "nn/Pygeon/models";
    cfg.data_dir = "nn/Pygeon/data";
    cfg.pretrained = "dummy";
    cfg.image_file = "all_zero";
    cfg.label_file = "all_zero";

#if MODELOWNER != -1 || DATAOWNER != -1  // If actual data is used, load paths from environment variables
    cfg.save_dir = std::getenv("MODEL_DIR") != NULL ? std::getenv("MODEL_DIR") : cfg.save_dir;
    cfg.data_dir = std::getenv("DATA_DIR") != NULL ? std::getenv("DATA_DIR") : cfg.data_dir;
    cfg.pretrained = std::getenv("MODEL_FILE") != NULL ? std::getenv("MODEL_FILE") : cfg.pretrained;
    cfg.image_file = std::getenv("SAMPLES_FILE") != NULL ? std::getenv("SAMPLES_FILE") : cfg.image_file;
    cfg.label_file = std::getenv("LABELS_FILE") != NULL ? std::getenv("LABELS_FILE") : cfg.label_file;

    // Print out the loaded configuration values
    print_init("Mode: " + cfg.mode);
    print_init("Save Directory: " + cfg.save_dir);
    print_init("Data Directory: " + cfg.data_dir);
    print_init("Pretrained Model: " + cfg.pretrained);
    print_init("Image File: " + cfg.image_file);
    print_init("Label File: " + cfg.label_file);
#endif

    cfg.batch = NUM_INPUTS * (BASE_DIV);

#if DATAOWNER == -1
    print_online("No Dataowner specified. Loading dummy data...");
    auto test_Y = read_dummy_labels(n_test);
    auto test_X = read_dummy_images(n_test, ch, h, w);
#else
#if PSELF == DATAOWNER
    print_online("Reading dataset from file...");
    string path = cfg.data_dir + "/" + cfg.image_file;
    auto test_X = read_custom_images(path, n_test, ch, h, w);
    print_online("Dataset imported.");
#else
    auto test_X = read_dummy_images(n_test, ch, h, w);
#endif
    string lpath = cfg.data_dir + "/" + cfg.label_file;
    auto test_Y = read_custom_labels(lpath, n_test);
#endif

#if JIT_VEC == 0
    MatX<modeltype> test_XX = test_X.unaryExpr([](LFLOATTYPE val) {
        modeltype tmp;
        tmp.template prepare_receive_and_replicate<DATAOWNER>(
            FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val));
        return tmp;
    });
    modeltype::communicate();
    for (int i = 0; i < test_XX.size(); i++)
    {
        test_XX(i).template complete_receive_from<DATAOWNER>();
    }
    print_online("Received Secret Share of Dataset");
#endif

#if JIT_VEC == 0
    DataLoader<modeltype> test_loader;
    test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);
#else
    DataLoader<LFLOATTYPE> test_loader;
    test_loader.load(test_X, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);
#endif

    // === Share Model Parameters ===

    print_online("Compiling model...");
    model.compile({cfg.batch / (BASE_DIV), ch, h, w});

    print_online("Loading model Parameters...");
#if PSELF == MODELOWNER
    print_online("Loading model parameters from file...");
#endif
#if MODELOWNER != -1
    model.template load<MODELOWNER>(cfg.save_dir, cfg.pretrained);
#else
    print_online("No Modelowner specified. Loading dummy parameters...");
#endif
    print_online("Received Secret Share of Model Parameters.");

    // === Inference ===

    model.evaluate(test_loader);
}

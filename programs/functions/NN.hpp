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

template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model)
{
	if (cfg.model == "lenet5") {
		for (int i = 0; i < 6; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Conv2d<T>(1, 6, 5, 1, 2, true, cfg.init));
				}
				else {
					model.add(new Conv2d<T>(6, 16, 5, 1, 0, true, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm2d<T>);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
				if (cfg.pool == "max") {
					model.add(new MaxPool2d<T>(2, 2));
				}
				else {
					model.add(new AvgPool2d<T>(2, 2));
				}
			}
			else if (i == 2) {
				model.add(new Flatten<T>);
			}
			else if (i < 5) {
				if (i == 3) {
					model.add(new Linear<T>(400, 120, cfg.init));
				}
				else {
					model.add(new Linear<T>(120, 84, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(84, 10, cfg.init));
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax<T>);
				}
				/* else { */
				/* 	model.add(new Sigmoid<T>); */
				/* } */
			}
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Linear<T>(784, 500, cfg.init));
				}
				else {
					model.add(new Linear<T>(500, 150, cfg.init));
				}
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(150, 10, cfg.init));
				if (cfg.use_batchnorm) {
					model.add(new BatchNorm1d<T>);
				}
				/* if (cfg.loss == "cross_entropy") { */
				/* 	model.add(new Softmax<T>); */
				/* } */
				/* else { */
				/* 	model.add(new Sigmoid<T>); */
				/* } */
			}
		}
	}
}
    /* template<typename Share> */
/* void inference(DATATYPE* res) */
/* { */
    /* /1* using DATATYPE = uint32_t; *1/ */
    /* using FLOATTYPE = float; */
    /* using SHARETYPE = Wrapper<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL, DATATYPE>; */
    /* using S = XOR_Share<DATATYPE, Share>; */
    /* using A = Additive_Share<DATATYPE, Share>; */
    /* using Bitset = sbitset_t<BITLENGTH, S>; */
    /* using sint = sint_t<A>; */
/* #if BASETYPE == 0 */
    /* using modeltype = A; */
/* #else */
    /* using modeltype = sint; */
/* #endif */
    /* /1* const int parallel_factor = 1; *1/ */
    /* /1* using cleartype = k_clear<A>; *1/ */

    /* /1* using Sharetype = Wrapper<DATATYPE>; *1/ */
    /* /1* using F = FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE> ; *1/ */
	/* Config cfg; */
   
    /* // Array of string literals */
    /* const char* literals[] = {"simplenn", "--mode=test", "--save_dir=./SimpleNN/model_zoo", "--pretrained=lenet5.pth"}; */

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

	/* int n_test = NUM_INPUTS*BASE_DIV, ch = 1, h = 28, w = 28; */
	
    /* MatX<float> test_X; */
	/* VecXi test_Y; */
/* #if IS_TRAINING == 1 */
	/* MatX<float> train_X; */
    /* VecXi train_Y; */
    /* int n_train = 60000; */
/* #endif */


/* #if JIT_VEC == 0 */
/* #if IS_TRAINING == 1 */
	/* DataLoader<modeltype> train_loader; */
/* #endif */
	/* DataLoader<modeltype> test_loader; */
/* #else */
/* #if IS_TRAINING == 1 */
	/* DataLoader<float> train_loader; */
/* #endif */
    /* DataLoader<float> test_loader; */
/* #endif */

/* #if IS_TRAINING == 1 */
		/* train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train); */
		/* train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train); */
    /*     MatX<A> train_XX = train_X.unaryExpr([](float val) { */ 
    /*             A tmp; */
    /*             tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); */
    /*             return tmp; */
    /*     /1* MatX<modeltype> train_XX = train_X.unaryExpr([](float val) { *1/ */ 
    /*     /1*         modeltype tmp; *1/ */
    /*     /1*         tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); *1/ */
    /*     /1*         return tmp; *1/ */
    /* /1* return sint(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); *1/ */
/* }); */

    /*             modeltype::communicate(); */
    /*         for (int i = 0; i < train_XX.size(); i++) { */
    /*             train_XX(i).template complete_receive_from<P_0>(); */
    /*         } */

		/* /1* train_loader.load(train_XX, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train); *1/ */
/* #endif */
    /* std::cout << "Reading MNIST test data..." << std::endl; */
	/* test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
	/* test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test); */
    /* /1* std::cout << "rows: " << test_X.rows() << std::endl; *1/ */
    /* /1* std::cout << "cols: " << test_X.cols() << std::endl; *1/ */
    /* /1* MatX<modeltype> test_XX(test_X.rows()/DATTYPE, test_X.cols()); *1/ */
    /* /1* for (int j = 0; j < test_X.cols(); j++) { *1/ */
    /* /1*     for (int i = 0; i < test_X.rows(); i+=DATTYPE) { *1/ */
    /* /1*         if(i+DATTYPE > test_X.rows()) { *1/ */
    /* /1*             break; // do not process leftovers *1/ */
    /* /1*         } *1/ */
    /* /1*     alignas(sizeof(DATATYPE)) UINT_TYPE tmp[DATTYPE]; *1/ */
    /* /1*     alignas(sizeof(DATATYPE)) DATATYPE tmp2[BITLENGTH]; *1/ */
    /* /1*     for (int k = 0; k < DATTYPE; ++k) { *1/ */
    /* /1*         tmp[k] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(test_X(i+k, j)); *1/ */
    /* /1*     } *1/ */
    /* /1*     orthogonalize_arithmetic(tmp, tmp2); *1/ */
    /* /1*     test_XX(i / DATTYPE, j).template prepare_receive_from<P_0>(tmp2); *1/ */
    /* /1*     } *1/ */
    /* /1* } *1/ */
    /* /1* modeltype::communicate(); *1/ */
    /* /1* for (int j = 0; j < test_XX.cols(); ++j) { *1/ */
    /* /1*     for (int i = 0; i < test_XX.rows(); ++i) { *1/ */
    /* /1*         test_XX(i, j).template complete_receive_from<P_0>(); *1/ */
    /* /1*     } *1/ */
    /* /1* } *1/ */
    /* /1* std::cout << "rows: " << test_XX.rows() << std::endl; *1/ */
    /* /1* std::cout << "cols: " << test_XX.cols() << std::endl; *1/ */

    /* /1* MatX<modeltype> test_XX = test_X.unaryExpr([](float val) { *1/ */ 
    /* /1*             modeltype tmp; *1/ */
    /* /1*             tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); *1/ */
    /* /1*             modeltype::communicate(); *1/ */
    /* /1*             tmp.template complete_receive_from<P_0>(); *1/ */
    /* /1*             return tmp; *1/ */

    /* /1* /2* return sint(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); *2/ *1/ */
    /* /1* }); *1/ */

    /* /1* MatX<modeltype> test_XX(test_X.rows()/DATTYPE, test_X.cols()); *1/ */
    /* /1* for (int j = 0; j < test_X.cols(); j++) { *1/ */
    /* /1*     for (int i = 0; i < test_X.rows(); i+=DATTYPE) { *1/ */
    /* /1*         if(i+DATTYPE > test_X.rows()) { *1/ */
    /* /1*             break; // do not process leftovers *1/ */
    /* /1*         } *1/ */
    /* /1*     alignas(sizeof(DATATYPE)) UINT_TYPE tmp[DATTYPE]; *1/ */
    /* /1*     alignas(sizeof(DATATYPE)) DATATYPE tmp2[BITLENGTH]; *1/ */
    /* /1*     for (int k = 0; k < DATTYPE; ++k) { *1/ */
    /* /1*         tmp[k] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(test_X(i+k, j)); *1/ */
    /* /1*     } *1/ */
    /* /1*     orthogonalize_arithmetic(tmp, tmp2); *1/ */
    /* /1*     test_XX(i / DATTYPE, j).template prepare_receive_from<P_0>(tmp2); *1/ */
    /* /1*     } *1/ */
    /* /1* } *1/ */
    /* /1* modeltype::communicate(); *1/ */
    /* /1* for (int j = 0; j < test_XX.cols(); ++j) { *1/ */
    /* /1*     for (int i = 0; i < test_XX.rows(); ++i) { *1/ */
    /* /1*         test_XX(i, j).template complete_receive_from<P_0>(); *1/ */
    /* /1*     } *1/ */
    /* /1* } *1/ */
/* #if JIT_VEC == 0 */ 
    /*     MatX<modeltype> test_XX = test_X.unaryExpr([](float val) { */ 
    /*             modeltype tmp; */
    /*             tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); */
    /*             return tmp; */
    /* /1* return sint(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); *1/ */
/* }); */

    /*             modeltype::communicate(); */
    /*         for (int i = 0; i < test_XX.size(); i++) { */
    /*             test_XX(i).template complete_receive_from<P_0>(); */
    /*         } */
    /* /1* MatX<modeltype> test_XX = test_X.unaryExpr([](float val) { *1/ */ 
    /* /1*             modeltype tmp; *1/ */
    /* /1*             tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); *1/ */
    /* /1*             modeltype::communicate(); *1/ */
    /* /1*             tmp.template complete_receive_from<P_0>(); *1/ */
    /* /1*             return tmp; *1/ */
    /* /1* }); *1/ */
/* #endif */
	/* /1* test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test); *1/ */
/* #if JIT_VEC == 0 */ 
	/* test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test); */
/* #else */
	/* test_loader.load(test_X, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test); */
/* #endif */

    /* std::cout << "Dataset loaded." << std::endl; */

    /* SimpleNN<modeltype> model; */
	/* load_model(cfg, model); */
	/* /1* model.add(new Conv2d<SHARETYPE>(1, 6, 5, 2, cfg.init)); *1/ */
    /* /1* model.add(new ReLU<SHARETYPE>); *1/ */
    /* /1* model.add(new MaxPool2d<SHARETYPE>(2, 2)); *1/ */
    /* /1* model.add(new Flatten<SHARETYPE>); *1/ */
    /* /1* model.add(new Linear<SHARETYPE>(400, 120, cfg.init)); *1/ */
	/* /1* model.add(new Softmax<SHARETYPE>); *1/ */

    /* /1* model.add(new AvgPool2d<SHARETYPE>(2, 2)); *1/ */
    /* /1* model.add(new BatchNorm2d<SHARETYPE>); *1/ */

	/* cout << "Model construction completed." << endl; */

    /* /1* if (cfg.mode == "train") { *1/ */
    /*     /1* if (cfg.loss == "cross_entropy") { *1/ */
    /*     /1*     model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new CrossEntropyLoss<sint>); *1/ */
    /*     /1* } *1/ */
    /*     /1* else { *1/ */
    /*     /1*     model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new MSELoss<sint>); *1/ */
    /*     /1* } *1/ */
    /*     /1* model.fit(train_loader, cfg.epoch, test_loader); *1/ */
    /*     /1* model.save("./model_zoo", cfg.model + ".pth"); *1/ */
    /* /1* } *1/ */
    
    /* /1* else { *1/ */
    /*     model.compile({ cfg.batch/(BASE_DIV), ch, h, w }); */
    /*     std::cout << "Loading Model Parameters..." << std::endl; */
    /*     model.template load<P_0>(cfg.save_dir, cfg.pretrained); */
    /*     model.evaluate(test_loader); */
    /* /1* } *1/ */

	/* /1* return 0; *1/ */
/* } */

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


#if FUNCTION_IDENTIFIER == 60
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 61
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 62 
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 10;
    auto model = DRD_C100_230K<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 63
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 64
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 65
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 224, w = 224, num_classes = 1000;
    auto model = DRD_C100_230K<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 66
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 100;
    auto model = VGG<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 67
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 100;
    auto model = ResNet50<modeltype>(num_classes);
#elif FUNCTION_IDENTIFIER == 68
    int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 100;
    auto model = DRD_C100_230K<modeltype>(num_classes);
#endif
    
    /* using DATATYPE = uint32_t; */
    /* const int parallel_factor = 1; */
    /* using cleartype = k_clear<A>; */

    /* using Sharetype = Wrapper<DATATYPE>; */
    /* using F = FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE> ; */
	Config cfg;
    cfg.mode = "test";
    cfg.save_dir = "./SimpleNN/model_zoo";
    cfg.data_dir = "./SimpleNN/dataset";
    /* cfg.pretrained = "model_DRD_C100_230K.bin"; */
    /* cfg.pretrained = "resnet50_cifar100.bin"; */
    cfg.pretrained = "dummy.dummy";
    /* cfg.pretrained = "vgg16_cifar.bin"; */
    /* cfg.pretrained = "lenet5.pth"; */
    cfg.image_file = "cifar10-test-images.bin";
    cfg.label_file = "cifar10-test-labels.bin";
    /* cfg.image_file = "CIFAR-100_test_images.bin"; */
    /* cfg.label_file = "CIFAR-100_test_labels.bin"; */
    cfg.batch = NUM_INPUTS*(BASE_DIV);

   
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
    auto test_X = read_dummy_images(n_test * ch, ch, h, w);
#else
#if PSELF == DATAOWNER
    print_online("Reading dataset from file...");
		/* auto test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
    string path = cfg.data_dir + "/" + cfg.image_file;
    auto test_X = read_custom_images(path, n_test * ch, ch, h, w);
    print_online("Dataset imported.");
#else
    auto test_X = read_dummy_images(n_test * ch, ch, h, w);
#endif
    string lpath = cfg.data_dir + "/" + cfg.label_file;
    auto test_Y = read_custom_labels(lpath, n_test);
#endif
/* #endif */
	/* test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
	/* test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test); */

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
        model.compile({ cfg.batch/(BASE_DIV), ch, h, w });
#if PSELF == MODELOWNER
        print_online("Loading model parameters from file...");
#endif
        model.template load<MODELOWNER>(cfg.save_dir, cfg.pretrained);
        print_online("Received Secret Share of Model Parameters.");
        model.evaluate(test_loader);
    /* } */

	/* return 0; */
}

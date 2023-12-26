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
					model.add(new Conv2d<T>(1, 6, 5, 1, 2, cfg.init));
				}
				else {
					model.add(new Conv2d<T>(6, 16, 5, 1, 0, cfg.init));
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

    template<typename Share>
void inference(DATATYPE* res)
{
	int n_test = NUM_INPUTS*BASE_DIV, ch = 3, h = 32, w = 32, num_classes = 100;
    
    /* using DATATYPE = uint32_t; */
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
    /* const int parallel_factor = 1; */
    /* using cleartype = k_clear<A>; */

    /* using Sharetype = Wrapper<DATATYPE>; */
    /* using F = FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE> ; */
	Config cfg;
    cfg.mode = "test";
    cfg.save_dir = "./SimpleNN/model_zoo";
    /* cfg.pretrained = "model_DRD_C100_230K.bin"; */
    /* cfg.pretrained = "resnet50_cifar100.bin"; */
    cfg.pretrained = "dummy.dummy";
    cfg.batch = NUM_INPUTS*BASE_DIV;

   
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
/* #if PARTY == DATAOWNER */
/*     auto test_X = read_custom_images("./dataset/CIFAR-100_test_images.bin", n_test * ch, ch, h, w); */
/*     auto test_Y = read_custom_labels("./dataset/CIFAR-100_test_labels.bin", n_test); */
/* #else */
#if PARTY == MODELOWNER
    print_online("Reading dataset from file...\n ");
#endif

    auto test_X = read_dummy_images(n_test * ch, ch, h, w);
    auto test_Y = read_dummy_labels(n_test);
/* #endif */
	/* test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
	/* test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test); */
#if PARTY == MODELOWNER
    print_online("Dataset imported.\n ");
#endif

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
    ResNet<modeltype> model = ResNet50<modeltype>(num_classes);
    /* SimpleNN<modeltype> model; */
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
#if PARTY == MODELOWNER
        print_online("Loading model parameters from file...\n ");
#endif
        model.template load<MODELOWNER>(cfg.save_dir, cfg.pretrained);
        print_online("Received Secret Share of Model Parameters.");
        model.evaluate(test_loader);
    /* } */

	/* return 0; */
}

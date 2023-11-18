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
#include "headers/config.h"

#include "Relu.hpp"
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
					model.add(new Conv2d<T>(1, 6, 5, 2, cfg.init));
				}
				else {
					model.add(new Conv2d<T>(6, 16, 5, 0, cfg.init));
				}
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm2d<T>); */
				/* } */
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
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(84, 10, cfg.init));
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				/* if (cfg.loss == "cross_entropy") { */
				/* 	model.add(new Softmax<T>); */
				/* } */
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
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(150, 10, cfg.init));
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
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
    /* using DATATYPE = uint32_t; */
    using FLOATTYPE = float;
    using SHARETYPE = Wrapper<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL, DATATYPE>;
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;
    using modeltype = sint;

    /* using Sharetype = Wrapper<DATATYPE>; */
    /* using F = FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE> ; */
	Config cfg;
   
    // Array of string literals
    const char* literals[] = {"simplenn", "--mode=test", "--save_dir=./SimpleNN/model_zoo", "--pretrained=lenet5.pth", "--pool=avg"};

    // Number of arguments
    size_t numArgs = sizeof(literals) / sizeof(literals[0]);

    // Allocate an array of char* pointers
    char** argv = new char*[numArgs];

    // Copy each string literal into a new dynamically allocated string
    for (size_t i = 0; i < numArgs; ++i) {
        size_t length = std::strlen(literals[i]);
        argv[i] = new char[length + 1];  // +1 for the null terminator
        std::strcpy(argv[i], literals[i]);
    } 
    int argc = 5;
	cfg.parse(argc, argv);
	cfg.print_config();

	int n_train = 60000, n_test = 10000, ch = 1, h = 28, w = 28;

	MatX<float> train_X, test_X;

	VecXi train_Y, test_Y;

	DataLoader<modeltype> train_loader, test_loader;

	if (cfg.mode == "train") {
		train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train);
		train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train);
        MatX<modeltype> train_XX = train_X.unaryExpr([](float val) { 
                modeltype tmp;
                tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val));
                modeltype::communicate();
                tmp.template complete_receive_from<P_0>();
                return tmp;
    /* return sint(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); */
});

		train_loader.load(train_XX, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train);
	}

    std::cout << "Reading MNIST test data..." << std::endl;
	test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test);
	test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test);
    
    MatX<modeltype> test_XX = test_X.unaryExpr([](float val) { 
                modeltype tmp;
                tmp.template prepare_receive_and_replicate<P_0>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val));
                modeltype::communicate();
                tmp.template complete_receive_from<P_0>();
                return tmp;

    /* return sint(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(val)); */
    });
	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

    std::cout << "Dataset loaded." << std::endl;

    SimpleNN<modeltype> model;
	load_model(cfg, model);
	/* model.add(new Conv2d<SHARETYPE>(1, 6, 5, 2, cfg.init)); */
    /* model.add(new ReLU<SHARETYPE>); */
    /* model.add(new MaxPool2d<SHARETYPE>(2, 2)); */
    /* model.add(new Flatten<SHARETYPE>); */
    /* model.add(new Linear<SHARETYPE>(400, 120, cfg.init)); */
	/* model.add(new Softmax<SHARETYPE>); */

    /* model.add(new AvgPool2d<SHARETYPE>(2, 2)); */
    /* model.add(new BatchNorm2d<SHARETYPE>); */

	cout << "Model construction completed." << endl;

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
        model.compile({ cfg.batch, ch, h, w });
        std::cout << "Loading Model Parameters..." << std::endl;
        model.template load<P_0>(cfg.save_dir, cfg.pretrained);
        model.evaluate(test_loader);
    /* } */

	/* return 0; */
}

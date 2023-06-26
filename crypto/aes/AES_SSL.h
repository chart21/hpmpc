#ifndef __AES_NI_H__
#define __AES_NI_H__
#include "openssl/aes.h"
#include <openssl/evp.h>
//macros


#define DO_ENC_BLOCK(m,k) \
	do{\
        int out_len;\
        EVP_EncryptUpdate(k, m, &out_len, m, 16);\
    }while(0)
#endif

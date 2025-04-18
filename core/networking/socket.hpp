#pragma once
#include "../../config.h"
#include "../include/pch.h"
#if USE_SSL == 1
#include <openssl/err.h>
#include <openssl/ssl.h>
#endif

// A simple class for representing a socket
class Socket
{

  private:
    // The socket descriptor
    int sock_;

#if USE_SSL == 1
    // The SSL structure
    SSL* ssl_;

    // The SSL context
    SSL_CTX* ssl_ctx_;

    // Private constructor for creating a Socket object from an SSL structure
    // Constructor (used by the Accept method)
    Socket(int sock, SSL* ssl) : sock_(sock), ssl_(ssl) { ssl_ctx_ = nullptr; }
#endif

    // Private constructor for creating a Socket object from a socket descriptor
    Socket(int sock) : sock_(sock) {}

  public:
    // Create a socket object
    Socket()
    {
        // Create a socket
        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ < 0)
        {
            throw std::runtime_error("Error creating socket");
        }
#if USE_SSL == 1
        ssl_ = nullptr;
        ssl_ctx_ = nullptr;
#endif
    }

    // Destroy the socket object
    ~Socket()
    {
        Close_Context();
        // Close the socket
    }

    // Bind the socket to a local address
    void Bind(int port)
    {
        // reuse the address
        int yes = 1;
        if (setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
        {
            perror("setsockopt");
            exit(1);
        }
        struct sockaddr_in address;
        int addrlen = sizeof(address);

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        while (::bind(sock_, (struct sockaddr*)&address, sizeof(address)) < 0)
        {
            /* throw std::runtime_error("Error connecting to remote server"); */
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - t1)
                    .count() > CONNECTION_TIMEOUT)
            {
                throw std::runtime_error("Timeout exceeded when trying to binding socket to local address on port " +
                                         std::to_string(port));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(CONNECTION_RETRY));
        }

    }  // Bind the socket to a local address

    // Listen for incoming connections
    void Listen(int backlog)
    {
        if (listen(sock_, backlog) < 0)
        {
            throw std::runtime_error("Error listening on socket");
        }
#if USE_SSL == 1
        // Create an SSL context
        ssl_ctx_ = SSL_CTX_new(TLS_server_method());
        if (ssl_ctx_ == nullptr)
        {
            throw std::runtime_error("Error creating SSL context");
        }

        // Configure the SSL context
        if (SSL_CTX_use_certificate_file(ssl_ctx_, "ssl_certificates/server.crt", SSL_FILETYPE_PEM) != 1)
        {
            throw std::runtime_error("Error loading certificate from file");
        }
        if (SSL_CTX_use_PrivateKey_file(ssl_ctx_, "ssl_certificates/server.key", SSL_FILETYPE_PEM) != 1)
        {
            throw std::runtime_error("Error loading private key from file");
        }

        signal(SIGPIPE, SIG_IGN);

        // Set session ID context
        /* const unsigned char *sid_ctx = reinterpret_cast<const unsigned char *>("MySessionIDContext"); */
        /* SSL_CTX_set_session_id_context(ssl_ctx_, sid_ctx, strlen(reinterpret_cast<const char *>(sid_ctx))); */

#endif
    }

    // Connect to a remote server
    void Connect(const std::string& addr, int port)
    {
#if PRINT == 1
        /* std::cout << "connecting to ip addr at port" << addr << " " << port << std::endl; */
#endif
        sockaddr_in addr_in;
        std::memset(&addr_in, 0, sizeof(addr_in));
        addr_in.sin_family = AF_INET;
        addr_in.sin_port = htons(port);
        if (inet_aton(addr.c_str(), &addr_in.sin_addr) == 0)
        {
            throw std::runtime_error("Invalid address: " + addr);
        }
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        while (connect(sock_, reinterpret_cast<sockaddr*>(&addr_in), sizeof(addr_in)) < 0)
        {
            /* throw std::runtime_error("Error connecting to remote server"); */
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - t1)
                    .count() > CONNECTION_TIMEOUT)
            {
                throw std::runtime_error("Timeout exceeded while connecting to server on port " + std::to_string(port));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(CONNECTION_RETRY));
        }
#if USE_SSL == 1
        // Create an SSL context
        ssl_ctx_ = SSL_CTX_new(SSLv23_client_method());
        if (ssl_ctx_ == nullptr)
        {
            throw std::runtime_error("Error creating SSL context");
        }

        // Create an SSL structure
        ssl_ = SSL_new(ssl_ctx_);
        if (ssl_ == nullptr)
        {
            throw std::runtime_error("Error creating SSL structure");
        }

        // Connect the SSL structure to the socket
        if (SSL_set_fd(ssl_, sock_) != 1)
        {
            throw std::runtime_error("Error connecting SSL structure to socket");
        }

        // Perform the SSL handshake
        if (SSL_connect(ssl_) != 1)
        {
            ERR_print_errors_fp(stderr);
            /* throw std::runtime_error("Error connecting to remote server"); */
            throw std::runtime_error("Error performing SSL handshake on port " + std::to_string(port));
        }
#endif
    }

    // Accept an incoming connection
    Socket Accept()
    {
        int client_sock;
        sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
#if USE_SSL == 1
        // Accept the incoming connection and create an SSL structure
        SSL* ssl = nullptr;
        if ((client_sock = accept(sock_, reinterpret_cast<sockaddr*>(&client_addr), &addr_len)) < 0)
        {
            throw std::runtime_error("Error accepting connection");
        }
        ssl = SSL_new(ssl_ctx_);
        SSL_set_fd(ssl, client_sock);
        if (SSL_accept(ssl) <= 0)
        {
            throw std::runtime_error("Error performing SSL handshake");
        }
        // Return the new socket
        return Socket(client_sock, ssl);
#else
        // Accept the incoming connection
        if ((client_sock = accept(sock_, reinterpret_cast<sockaddr*>(&client_addr), &addr_len)) < 0)
        {
            throw std::runtime_error("Error accepting connection");
        }
        // Return the new socket
        return Socket(client_sock);

#endif
    }

    void Send_all(char* buffer, int64_t* len)
    {
        int64_t total = 0;         // how many bytes we've sent
        int64_t bytesleft = *len;  // how many we have left to send
        int64_t n_sent;
        while (total < *len)
        {
            n_sent = Send(buffer + total, bytesleft);
            if (n_sent == -1)
            {
                break;
            }
            total += n_sent;
            bytesleft -= n_sent;
        }
        *len = total;  // return number actually sent here
    }

    // Send data to the client
    int64_t Send(char* buffer, int64_t size)
    {
#if USE_SSL == 1
        int64_t num_bytes = SSL_write(ssl_, buffer, size);
#else
        int64_t num_bytes = send(sock_, buffer, size, 0);
#endif
        if (num_bytes < 0)
        {
            throw std::runtime_error("Error sending data");
        }
        return num_bytes;
    }

    void Receive_all(char* buffer, int64_t* len)
    {

        int64_t total = 0;         // how many bytes we've received
        int64_t bytesleft = *len;  // how many we have left to receive
        int64_t n_rec;
        while (total < *len)
        {
            n_rec = Receive(buffer + total, bytesleft);
            if (n_rec == -1)
            {
                break;
            }
            total += n_rec;
            bytesleft -= n_rec;
        }
        *len = total;  // return number actually received here
    }

    // Receive data from the client
    int64_t Receive(char* buffer, int64_t size)
    {
#if USE_SSL == 1
        int64_t num_bytes = SSL_read(ssl_, buffer, size);
#else
        int64_t num_bytes = recv(sock_, buffer, size, 0);
#endif
        if (num_bytes < 0)
        {
            throw std::runtime_error("Error receiving data");
        }
        return num_bytes;
    }

    void Close_Context()
    {
        if (sock_ != -1)
        {
            close(sock_);
            sock_ = -1;
        }
#if USE_SSL == 1
        // Free the SSL structure
        if (ssl_ != nullptr)
        {
            if (SSL_shutdown(ssl_) == 0)
            {
                if (SSL_shutdown(ssl_) < 0)
                {

                    std::cout << "P" << PARTY << ": SSL_shutdown failed" << std::endl;
                }
            }
            SSL_free(ssl_);
            ssl_ = nullptr;
        }

        // Free the SSL context
        if (ssl_ctx_ != nullptr)
        {
            SSL_CTX_free(ssl_ctx_);
            ssl_ctx_ = nullptr;
        }
#endif
    }
};

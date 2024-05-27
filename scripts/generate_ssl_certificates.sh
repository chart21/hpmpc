#!/bin/bash

# Directory to store the SSL certificates
CERT_DIR="ssl_certificates"

# Create the directory if it doesn't exist
mkdir -p $CERT_DIR

# Generate a new private key without password protection
openssl genpkey -algorithm RSA -out $CERT_DIR/server.key

# Generate a certificate signing request (CSR)
openssl req -new -key $CERT_DIR/server.key -out $CERT_DIR/server.csr -subj "/C=DE/ST=Some-State/O=Internet Widgits Pty Ltd"

# Sign the certificate
openssl x509 -req -days 36500 -in $CERT_DIR/server.csr -signkey $CERT_DIR/server.key -out $CERT_DIR/server.crt

# Clean up the CSR
rm $CERT_DIR/server.csr

echo "New private key and certificate have been generated and saved in the $CERT_DIR directory."


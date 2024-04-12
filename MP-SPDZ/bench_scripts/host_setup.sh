#!/bin/bash

# Global setup-script running locally on experiment server. 
# Initializing the experiment server

# exit on error
set -e             
# log every command
set -x                         

REPO=$(pos_get_variable repo_hpmpc --from-global)
REPO_COMMIT=$(pos_get_variable repo_hpmpc_commit --from-global)       
REPO_DIR=$(pos_get_variable repo_hpmpc_dir --from-global)
REPO2=$(pos_get_variable repo --from-global)
REPO2_DIR=$(pos_get_variable repo_dir --from-global)
REPO3=$(pos_get_variable repo_mpspdz --from-global)
#REPO_COMMIT=$(pos_get_variable repo_commit --from-global)       
REPO3_DIR=$(pos_get_variable repo_mpspdz_dir --from-global)

# check WAN connection, waiting helps in most cases
checkConnection() {
    address=$1
    i=0
    maxtry=5
    success=false
    while [ $i -lt $maxtry ] && ! $success; do
        success=true
        echo "____ping $1 try $i" >> pinglog_external
        ping -q -c 2 "$address" >> pinglog_external || success=false
        ((++i))
        sleep 2s
    done
    $success
}

checkConnection "mirror.lrz.de"
echo 'unattended-upgrades unattended-upgrades/enable_auto_updates boolean false' | debconf-set-selections
export DEBIAN_FRONTEND=noninteractive
apt update
apt install -y automake build-essential git libboost-dev libboost-thread-dev parted \
    libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm linux-cpupower \
    python3-pip time software-properties-common iperf3
# wget https://apt.llvm.org/llvm.sh
# chmod +x llvm.sh
# ./llvm.sh -y 15
# apt install -y clang-15 gcc-12 g++-12


echo 'deb http://deb.debian.org/debian testing main' > /etc/apt/sources.list.d/testing.list
apt update -y
apt install -y gcc-12 g++-12
apt install libeigen3-dev
# wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
# tar -xvf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
# mv clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04 /usr/local/llvm-16
# echo 'export PATH="/usr/local/llvm-16/bin:$PATH"' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH="/usr/local/llvm-16/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
# source ~/.bashrc

pip3 install -U numpy
checkConnection "github.com"

git clone "$REPO" "$REPO_DIR" # hpmcp
git clone "$REPO2" "$REPO2_DIR" # mpcbench
wget https://github.com/data61/MP-SPDZ/releases/download/v0.3.8/mp-spdz-0.3.8.tar.xz
tar -xf mp-spdz-0.3.8.tar.xz 
mv mp-spdz-0.3.8 "$REPO3_DIR" # MP-SPDZ


################################# mpcbench ####################################
cd "$REPO2_DIR" # mpcbench
# git checkout fix
cd ..

# load custom htop config
mkdir -p .config/htop
cp "$REPO2_DIR"/helpers/htoprc ~/.config/htop/

################################### hpmpc #####################################
cd "$REPO_DIR" # hpmpc

# use a stable state of the MP-Slice repo
###git checkout "$REPO_COMMIT"

# switch to fork
# git checkout extended
git checkout mp-spdz

git clone https://github.com/chart21/flexNN.git SimpleNN
cd SimpleNN
git checkout hpmpc
cd ..

# adjust script to specific needs
echo "wait" >> ./scripts/split-roles-3-execute.sh
echo "wait" >> ./scripts/split-roles-3to4-execute.sh
echo "wait" >> ./scripts/split-roles-4-execute.sh

echo "global setup successful"
 
################################## MP-SPDZ ####################################
cd "$REPO3_DIR"

cp "$REPO_DIR"/MP-SPDZ/Functions/bench/* ./Programs/Source/
cp "$REPO2_DIR"/experiments/mp-spdz/66_ThresholdSecurity/aes_128.txt ./Programs/Circuits/

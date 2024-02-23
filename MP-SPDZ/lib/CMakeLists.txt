set(VM_LIB machine)


add_library(${VM_LIB}
    Program.cpp
    Machine.cpp
    help/Util.cpp
    help/Input.cpp)

target_include_directories(${VM_LIB} PUBLIC ${CMAKE_SOURCE_DIR}/lib)


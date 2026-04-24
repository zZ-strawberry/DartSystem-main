cmake_minimum_required(VERSION 3.20)
project(drv_ws_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

set(CANDRV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../drv_ws/src/candrv)
set(DM_SDK_LIB ${CANDRV_ROOT}/lib/libdm_device.so)

add_library(
    drv_ws_bridge SHARED
    src/drv_ws_bridge.cpp
    ${CANDRV_ROOT}/src/candrv.cpp
    ${CANDRV_ROOT}/src/dameowcan.cpp
    ${CANDRV_ROOT}/src/socketcan.cpp
)

target_include_directories(
    drv_ws_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CANDRV_ROOT}/include
    ${LIBUSB_INCLUDE_DIRS}
)

target_link_directories(drv_ws_bridge PRIVATE ${LIBUSB_LIBRARY_DIRS})
target_link_libraries(
    drv_ws_bridge PRIVATE
    ${DM_SDK_LIB}
    ${LIBUSB_LIBRARIES}
    Threads::Threads
)

set_target_properties(
    drv_ws_bridge PROPERTIES
    BUILD_RPATH "${CANDRV_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN"
)

add_custom_command(
    TARGET drv_ws_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DM_SDK_LIB}
            $<TARGET_FILE_DIR:drv_ws_bridge>/libdm_device.so
)

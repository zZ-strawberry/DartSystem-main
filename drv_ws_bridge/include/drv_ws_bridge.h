#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* drv_ws_handle_t;

drv_ws_handle_t drv_ws_open(
    const char* driver,
    const char* selector,
    int bus_mode,
    uint32_t baud_rate,
    uint32_t data_baud_rate,
    uint32_t device_type,
    uint8_t channel,
    char* error_buffer,
    size_t error_buffer_size
);

int drv_ws_send(
    drv_ws_handle_t handle,
    uint32_t can_id,
    int is_extended,
    int is_fd,
    int brs,
    uint8_t len,
    const uint8_t* data,
    char* error_buffer,
    size_t error_buffer_size
);

void drv_ws_close(drv_ws_handle_t handle);

#ifdef __cplusplus
}
#endif

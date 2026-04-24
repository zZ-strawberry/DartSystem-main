#include "drv_ws_bridge.h"

#include "candrv.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

struct DriverHandle
{
    std::shared_ptr<candrv::ICanDriver> driver;
};

void write_error(const std::string& message, char* buffer, size_t buffer_size)
{
    if (!buffer || buffer_size == 0)
    {
        return;
    }
    const size_t copy_size = std::min(buffer_size - 1, message.size());
    std::memcpy(buffer, message.data(), copy_size);
    buffer[copy_size] = '\0';
}

candrv::DriverType parse_driver(const char* driver)
{
    const std::string name = driver ? driver : "";
    if (name == "dameow")
    {
        return candrv::DriverType::DaMeow;
    }
    if (name == "socketcan")
    {
        return candrv::DriverType::SocketCAN;
    }
    throw std::invalid_argument("Unsupported driver: " + name);
}

candrv::CanBusMode parse_bus_mode(int bus_mode)
{
    return bus_mode == 1 ? candrv::CanBusMode::CANFD : candrv::CanBusMode::CAN;
}

} // namespace

extern "C"
{

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
)
{
    try
    {
        const auto driver_type = parse_driver(driver);
        auto instance = candrv::create_driver(driver_type);

        candrv::CanConfig config;
        config.type = driver_type;
        config.interface_name = selector ? selector : "";
        config.bus_mode = parse_bus_mode(bus_mode);
        config.baud_rate = baud_rate;
        config.data_baud_rate = data_baud_rate;
        config.device_type = device_type;
        config.channel = channel;

        if (!instance->open(config))
        {
            write_error(instance->get_error(), error_buffer, error_buffer_size);
            return nullptr;
        }

        auto* handle = new DriverHandle();
        handle->driver = std::move(instance);
        write_error("", error_buffer, error_buffer_size);
        return reinterpret_cast<drv_ws_handle_t>(handle);
    }
    catch (const std::exception& exc)
    {
        write_error(exc.what(), error_buffer, error_buffer_size);
        return nullptr;
    }
}

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
)
{
    auto* driver_handle = reinterpret_cast<DriverHandle*>(handle);
    if (!driver_handle || !driver_handle->driver)
    {
        write_error("Invalid drv_ws handle", error_buffer, error_buffer_size);
        return 0;
    }
    if (len > 64)
    {
        write_error("CAN frame length exceeds 64 bytes", error_buffer, error_buffer_size);
        return 0;
    }
    if (len > 0 && data == nullptr)
    {
        write_error("Payload pointer is null", error_buffer, error_buffer_size);
        return 0;
    }

    candrv::CanFrame frame;
    frame.can_id = can_id;
    frame.len = len;
    frame.is_extended = (is_extended != 0);
    frame.is_fd = (is_fd != 0);
    frame.brs = (brs != 0);
    if (len > 0)
    {
        std::memcpy(frame.data, data, len);
    }

    if (!driver_handle->driver->send(frame))
    {
        write_error(driver_handle->driver->get_error(), error_buffer, error_buffer_size);
        return 0;
    }

    write_error("", error_buffer, error_buffer_size);
    return 1;
}

void drv_ws_close(drv_ws_handle_t handle)
{
    auto* driver_handle = reinterpret_cast<DriverHandle*>(handle);
    if (!driver_handle)
    {
        return;
    }
    if (driver_handle->driver)
    {
        driver_handle->driver->close();
        driver_handle->driver.reset();
    }
    delete driver_handle;
}

} // extern "C"

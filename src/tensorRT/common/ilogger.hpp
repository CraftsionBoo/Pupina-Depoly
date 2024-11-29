#ifndef __I_LOGGER_HPP__
#define __I_LOGGER_HPP__

#include <stdio.h>
#include <string>

namespace Taurus
{
    namespace iLogger
    {
        enum class LogLevel : int
        {
            Debug = 5,
            Verbose = 4,
            Info = 3,
            Warning = 2,
            Error = 1,
            Fatal = 0
        };

        #define INFOD(...) Taurus::iLogger::__log_func(__FILE__, __LINE__, Taurus::iLogger::LogLevel::Debug, __VA_ARGS__);
        #define INFOV(...) Taurus::iLogger::__log_func(__FILE__, __LINE__, Taurus::iLogger::LogLevel::Verbose, __VA_ARGS__);
        #define INFOI(...) Taurus::iLogger::__log_func(__FILE__, __LINE__, Taurus::iLogger::LogLevel::Info, __VA_ARGS__);
        #define INFOW(...) Taurus::iLogger::__log_func(__FILE__, __LINE__, Taurus::iLogger::LogLevel::Warning, __VA_ARGS__);
        #define INFOE(...) Taurus::iLogger::__log_func(__FILE__, __LINE__, Taurus::iLogger::LogLevel::Error, __VA_ARGS__);
        #define INFOF(...) Taurus::iLogger::__log_func(__FILE__, __LINE__, Taurus::iLogger::LogLevel::Fatal, __VA_ARGS__);

        void set_log_save_directory(const std::string &logger_dir); // 设置日志存储路径
        void set_log_level(LogLevel level);                         // 设置日志等级
        LogLevel get_log_level();                                   // 获取日志等级

        void __log_func(const char *file, int line, LogLevel level, const char *fmt, ...);

    }; // iLogger

}; // Taurus

#endif
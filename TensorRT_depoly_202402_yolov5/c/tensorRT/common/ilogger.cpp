#include "ilogger.hpp"
#include "cv_cpp_utils.hpp"

#include <mutex>
#include <thread>
#include <memory>
#include <vector>
#include <future>     // For atomic
#include <functional> // Fro bind
#include <stdarg.h>   // For params

namespace Taurus
{
    namespace iLogger
    {
        using namespace cUtils;

        /** 便于保存文件时，单独启用线程进行日志保存处理 **/
        static struct Logger  
        {
            LogLevel level_{LogLevel::Info};
            std::mutex lock_;
            std::string directory_;
            std::atomic<bool> running_{false};
            std::shared_ptr<FILE> handler_;
            std::vector<std::string> cache_, local_;
            std::shared_ptr<std::thread> flush_thread_;
            bool logger_shutdown_{false};

            void write(const std::string &line)
            {
                std::lock_guard<std::mutex> l(lock_);
                if (logger_shutdown_)
                    return;

                if (!running_)
                {
                    if (flush_thread_)
                        return;

                    cache_.reserve(1000);
                    running_ = true;
                    flush_thread_.reset(new std::thread(std::bind(&Logger::worker, this)));
                }
                cache_.emplace_back(line);
            }

            void flush()
            {
                if (cache_.empty())
                    return;
                {
                    std::unique_lock<std::mutex> lk(lock_);
                    std::swap(local_, cache_);
                }

                if (!local_.empty() || !directory_.empty())
                {
                    auto now = time_now();
                    auto file = format2048("%s%s.log", directory_.c_str(), now.c_str());
                    if (!exists(file)) // create log files
                        handler_.reset(fopen_mkdirs(file, "wb"), fclose);
                    else if (!handler_) // add to files
                        handler_.reset(fopen_mkdirs(file, "a+"), fclose);

                    if (handler_)
                    {
                        for (auto &line : local_)
                            fprintf(handler_.get(), "%s\n", line.c_str());
                        fflush(handler_.get());
                        handler_.reset();
                    }
                }
                local_.clear();
            }

            void worker()
            {
                auto tick_now = timestamp_now();
                std::vector<std::string> local;
                while (running_)
                {
                    if (timestamp_now() - tick_now < 1000)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    tick_now = timestamp_now();
                    flush();
                }
                flush();
            }

            void set_save_directory(const std::string &logger_dir)
            {
                directory_ = logger_dir;
                if (directory_.empty())
                    directory_ = ".";

                if (directory_.back() not_eq '/')
                    directory_.push_back('/');
            }

            void set_log_level(LogLevel level)
            {
                level_ = level;
            }

            void close()
            {
                {
                    std::lock_guard<std::mutex> l(lock_);
                    if (logger_shutdown_)
                        return;
                    logger_shutdown_ = true;
                }

                if (!running_)
                    return;

                running_ = false;
                flush_thread_->join();
                flush_thread_.reset();
                handler_.reset();
            }

            virtual ~Logger()
            {
                close();
            }

        } __g_logger_;

        const char *level_string(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::Debug:
                return "debug";
            case LogLevel::Verbose:
                return "verbo";
            case LogLevel::Info:
                return "info";
            case LogLevel::Warning:
                return "warn";
            case LogLevel::Error:
                return "error";
            case LogLevel::Fatal:
                return "fatal";
            default:
                return "unknow";
            }
        }

        static void remove_color_for_saving_text(char *buffer)
        {
            //"\033[31m%s\033[0m"
            char *p = buffer;
            while (*p)
            {
                if (*p = 0x1B) // \033
                {
                    char np = *(p + 1);
                    if (np == '[')
                    {
                        // has token
                        char *t = p + 2; // color numbers
                        while (*t)
                        {
                            if (*t == 'm')
                            {
                                t = t + 1;
                                char *k = p;
                                while (*t)
                                    *k++ = *t++;
                                *k = 0;
                                break;
                            }
                            t++;
                        }
                    }
                }
                p++;
            }
        }

        void set_log_save_directory(const std::string &logger_dir)
        {
            __g_logger_.set_save_directory(logger_dir);
        }

        void set_log_level(LogLevel level)
        {
            __g_logger_.set_log_level(level);
        }

        LogLevel get_log_level()
        {
            return __g_logger_.level_;
        }

        void __log_func(const char *file, int line, LogLevel level, const char *fmt, ...)
        {
            if (level > __g_logger_.level_)
                return;

            va_list vl;
            va_start(vl, fmt);

            char buffer[2048];
            auto now = time_now();
            std::string filename = file_name(file, true);

            int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());
            if (level == LogLevel::Fatal or level == LogLevel::Error) // red
            {
                n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]", level_string(level));
            }
            else if (level == LogLevel::Warning) // yellow
            {
                n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]", level_string(level));
            }
            else if (level == LogLevel::Info) // purple
            {
                n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]", level_string(level));
            }
            else if (level == LogLevel::Verbose)
            {
                n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[34m%s\033[0m]", level_string(level));
            }
            else
            {
                n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
            }

            n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]:", filename.c_str(), line);
            vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

            if (level == LogLevel::Debug or level == LogLevel::Error)
            {
                fprintf(stderr, "%s\n", buffer);
            }
            else if (level == LogLevel::Warning)
            {
                fprintf(stdout, "%s\n", buffer);
            }
            else
            {
                fprintf(stdout, "%s\n", buffer);
            }

            if (!__g_logger_.directory_.empty())
            {
                remove_color_for_saving_text(buffer);
                __g_logger_.write(buffer);
                if (level == LogLevel::Fatal)
                    __g_logger_.flush();
            }

            if (level == LogLevel::Fatal || level == LogLevel::Error)
            {
                fflush(stdout);
                abort();
            }
        }

    }; // namespace iLogger
}; // namespace Taurus
#ifndef __CV_CPP_UTILS_HPP__
#define __CV_CPP_UTILS_HPP__

#include <string>
#include <vector>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <tuple>

namespace Taurus
{
    namespace cUtils
    {
        /******** mkdirs  *********/
        ///////////////////////////////////////////////////////////////
        bool mkdir(const std::string &path);
        bool mkdirs(const std::string &path);
        FILE *fopen_mkdirs(const std::string &path, const std::string &mode);
        std::string file_name(const std::string &path, bool include_suffix);
        std::string file_path_name(const std::string &path);
        
        bool exists(const std::string &files);
        std::vector<unsigned char> load_file(const std::string &file);
        std::vector<std::string> find_files(const std::string &directory, const std::string &filter,
                                            bool findDirectory = false, bool includeSubDirectory = false);
        bool pattern_match(const char *str, const char *matcher, bool ignore_case = true);
        bool rmtree(const std::string &directory, bool ignore_fail = false);
        bool isDirectory(const std::string &path);
        
        /******** times  *********/
        //////////////////////////////////////////////////////////////
        std::string date_now();
        long long timestamp_now();
        double timestamp_now_float();
        std::string time_now();

        /******** params  *********/
        ///////////////////////////////////////////////////////////////
        std::string format2048(const char *fmt, ...); // get buffer 2048

        /******** utils *********/
        ///////////////////////////////////////////////////////////////
        inline int upbound(int n, int align = 32){return (n + align - 1) / align * align;}
        std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
        std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
    }; // namespace cUtils
}; // Taurus

#endif
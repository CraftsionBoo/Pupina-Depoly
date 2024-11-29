#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int app_yolo_det();
int app_yolo_seg();
int app_depthanything_v2();


int main(int argc, char** argv)
{
    const char* method = "yolo_det";
    if(argc > 1)
    {
        method = argv[1];
    }
    if(strcmp(method, "yolo_det") == 0)
        app_yolo_det();
    else if(strcmp(method, "yolo_seg") == 0)
        app_yolo_seg();
    else if(strcmp(method, "depth") == 0)
        app_depthanything_v2();
    else
        INFOE("Model not recognized");
    return 0;
}
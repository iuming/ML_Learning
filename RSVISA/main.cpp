#include <iostream>
#include <visa.h>

int main() {
    ViStatus status;
    ViSession defaultRM, vi;
    char instrDescriptor[] = "TCPIP0::192.168.8.106::inst0::INSTR"; // 请替换为实际的VNA IP地址

    // 打开默认资源管理器
    status = viOpenDefaultRM(&defaultRM);
    if (status < VI_SUCCESS) {
        std::cerr << "无法打开资源管理器" << std::endl;
        return 1;
    }
//
//    // 打开与VNA的连接
//    status = viOpen(defaultRM, instrDescriptor, VI_NULL, VI_NULL, &vi);
//    if (status < VI_SUCCESS) {
//        std::cerr << "无法打开与VNA的连接" << std::endl;
//        viClose(defaultRM);
//        return 1;
//    }
//
//    // 发送获取Marker1频率的命令
//    char command[] = "CALCulate:MARKer1:X?";
//    char response[100];
//    status = viPrintf(vi, command);
//    status = viScanf(vi, "%s", response);
//
//    // 打印Marker1的频率
//    std::cout << "Marker1的频率是：" << response << std::endl;
//
//    // 关闭与VNA的连接
//    viClose(vi);
//    viClose(defaultRM);

    return 0;
}

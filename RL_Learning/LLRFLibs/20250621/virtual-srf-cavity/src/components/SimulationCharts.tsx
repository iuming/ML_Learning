import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { SimulationResult } from '@/lib/srf-physics';

interface SimulationChartsProps {
  data: SimulationResult | null;
  isLoading: boolean;
}

const SimulationCharts: React.FC<SimulationChartsProps> = ({ data, isLoading }) => {
  console.log('SimulationCharts rendered with data:', data, 'isLoading:', isLoading);
  
  if (isLoading) {
    return (
      <Card className="w-full h-96">
        <CardContent className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">正在计算仿真结果...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="w-full h-96">
        <CardContent className="flex items-center justify-center h-full">
          <div className="text-center text-gray-500">
            <p className="text-lg mb-2">暂无数据</p>
            <p className="text-sm">请调整参数并点击"开始仿真"</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // 准备图表数据
  const chartData = data.time.map((t, i) => ({
    time: t * 1000, // 转换为毫秒
    cavityVoltageAmp: data.cavityVoltageAmp[i],
    cavityVoltagePhase: data.cavityVoltagePhase[i],
    dynamicDetuning: data.dynamicDetuning[i],
    reflectedPower: data.reflectedPower[i],
    forwardPower: data.forwardPower[i],
    beamCurrent: data.beamCurrent[i] * 1000, // 转换为mA
    mechanicalDisplacement: data.mechanicalDisplacement[i]
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-medium">{`时间: ${(label as number).toFixed(2)} ms`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.name}: ${entry.value.toFixed(4)} ${entry.unit || ''}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full space-y-6">
      <Tabs defaultValue="cavity" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="cavity">腔体电压</TabsTrigger>
          <TabsTrigger value="detuning">失谐效应</TabsTrigger>
          <TabsTrigger value="power">功率分析</TabsTrigger>
          <TabsTrigger value="mechanical">机械振动</TabsTrigger>
        </TabsList>

        <TabsContent value="cavity" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* 腔体电压幅度 */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">腔体电压幅度</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: '时间 (ms)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '电压幅度 (V)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      dataKey="cavityVoltageAmp" 
                      stroke="#2563eb" 
                      strokeWidth={2}
                      dot={false}
                      name="腔体电压"
                      unit="V"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* 腔体电压相位 */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">腔体电压相位</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: '时间 (ms)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '相位 (度)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      dataKey="cavityVoltagePhase" 
                      stroke="#dc2626" 
                      strokeWidth={2}
                      dot={false}
                      name="电压相位"
                      unit="°"
                    />
                    <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="detuning" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* 动态失谐 */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">动态失谐效应</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: '时间 (ms)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '失谐 (Hz)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      dataKey="dynamicDetuning" 
                      stroke="#16a34a" 
                      strokeWidth={2}
                      dot={false}
                      name="动态失谐"
                      unit="Hz"
                    />
                    <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* 束流脉冲 */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">束流脉冲</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: '时间 (ms)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '束流 (mA)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      dataKey="beamCurrent" 
                      stroke="#ea580c" 
                      strokeWidth={2}
                      dot={false}
                      name="束流强度"
                      unit="mA"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="power" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">功率分析</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="time" 
                    label={{ value: '时间 (ms)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: '功率 (MW)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="forwardPower" 
                    stroke="#2563eb" 
                    strokeWidth={2}
                    dot={false}
                    name="前向功率"
                    unit="MW"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="reflectedPower" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    dot={false}
                    name="反射功率"
                    unit="MW"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mechanical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">机械振动位移</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="time" 
                    label={{ value: '时间 (ms)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: '位移 (nm)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="mechanicalDisplacement" 
                    stroke="#7c3aed" 
                    strokeWidth={2}
                    dot={false}
                    name="机械位移"
                    unit="nm"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SimulationCharts;

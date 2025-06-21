import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, RotateCcw } from 'lucide-react';
import { SRFParameters, MechanicalModes } from '@/lib/srf-physics';

interface ParameterControlsProps {
  parameters: SRFParameters;
  mechanicalModes: MechanicalModes;
  onParametersChange: (parameters: SRFParameters) => void;
  onMechanicalModesChange: (modes: MechanicalModes) => void;
  onReset: () => void;
  onSimulate: () => void;
  isSimulating: boolean;
}

const ParameterControls: React.FC<ParameterControlsProps> = ({
  parameters,
  mechanicalModes,
  onParametersChange,
  onMechanicalModesChange,
  onReset,
  onSimulate,
  isSimulating
}) => {
  const updateParameter = (key: keyof SRFParameters, value: number) => {
    onParametersChange({ ...parameters, [key]: value });
  };

  const updateMechanicalMode = (index: number, type: 'frequencies' | 'qualityFactors' | 'lorentzCoefficients', value: number) => {
    const newModes = { ...mechanicalModes };
    newModes[type][index] = value;
    onMechanicalModesChange(newModes);
  };

  const formatValue = (value: number, precision: number = 3): string => {
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return value.toFixed(precision);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>超导腔参数控制</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={onReset}>
              <RotateCcw className="w-4 h-4 mr-2" />
              重置
            </Button>
            <Button 
              onClick={onSimulate} 
              disabled={isSimulating}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {isSimulating ? '计算中...' : '开始仿真'}
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="basic">基本参数</TabsTrigger>
            <TabsTrigger value="advanced">高级参数</TabsTrigger>
            <TabsTrigger value="mechanical">机械模式</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 束流强度 */}
              <div className="space-y-3">
                <Label htmlFor="ib" className="text-sm font-medium">
                  束流强度 (A): {parameters.ib.toFixed(3)}
                </Label>
                <Slider
                  id="ib"
                  min={0}
                  max={0.02}
                  step={0.001}
                  value={[parameters.ib]}
                  onValueChange={(value) => updateParameter('ib', value[0])}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0.000</span>
                  <span>0.020</span>
                </div>
              </div>

              {/* 初始失谐 */}
              <div className="space-y-3">
                <Label htmlFor="dw0" className="text-sm font-medium">
                  初始失谐 (Hz): {parameters.dw0.toFixed(0)}
                </Label>
                <Slider
                  id="dw0"
                  min={-1000}
                  max={1000}
                  step={10}
                  value={[parameters.dw0]}
                  onValueChange={(value) => updateParameter('dw0', value[0])}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>-1000</span>
                  <span>+1000</span>
                </div>
              </div>

              {/* 驱动幅度 */}
              <div className="space-y-3">
                <Label htmlFor="driveAmplitude" className="text-sm font-medium">
                  驱动幅度: {parameters.driveAmplitude.toFixed(1)}
                </Label>
                <Slider
                  id="driveAmplitude"
                  min={0.1}
                  max={3.0}
                  step={0.1}
                  value={[parameters.driveAmplitude]}
                  onValueChange={(value) => updateParameter('driveAmplitude', value[0])}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0.1</span>
                  <span>3.0</span>
                </div>
              </div>

              {/* 仿真时间 */}
              <div className="space-y-3">
                <Label htmlFor="simulationTime" className="text-sm font-medium">
                  仿真时间 (s): {parameters.simulationTime.toFixed(2)}
                </Label>
                <Slider
                  id="simulationTime"
                  min={0.05}
                  max={0.5}
                  step={0.01}
                  value={[parameters.simulationTime]}
                  onValueChange={(value) => updateParameter('simulationTime', value[0])}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0.05</span>
                  <span>0.50</span>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 有载品质因数 */}
              <div className="space-y-3">
                <Label htmlFor="QL" className="text-sm font-medium">
                  有载品质因数 QL: {formatValue(parameters.QL)}
                </Label>
                <Slider
                  id="QL"
                  min={1e5}
                  max={1e7}
                  step={1e5}
                  value={[parameters.QL]}
                  onValueChange={(value) => updateParameter('QL', value[0])}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>100K</span>
                  <span>10M</span>
                </div>
              </div>

              {/* 耦合系数 */}
              <div className="space-y-3">
                <Label htmlFor="beta" className="text-sm font-medium">
                  耦合系数 β: {formatValue(parameters.beta)}
                </Label>
                <Slider
                  id="beta"
                  min={1e3}
                  max={1e5}
                  step={1e3}
                  value={[parameters.beta]}
                  onValueChange={(value) => updateParameter('beta', value[0])}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>1K</span>
                  <span>100K</span>
                </div>
              </div>

              {/* R/Q 阻抗 */}
              <div className="space-y-3">
                <Label htmlFor="roQ" className="text-sm font-medium">
                  R/Q 阻抗 (Ω): {parameters.roQ.toFixed(0)}
                </Label>
                <Input
                  id="roQ"
                  type="number"
                  value={parameters.roQ}
                  onChange={(e) => updateParameter('roQ', parseFloat(e.target.value) || 0)}
                  className="w-full"
                />
              </div>

              {/* 射频频率 */}
              <div className="space-y-3">
                <Label htmlFor="f0" className="text-sm font-medium">
                  射频频率 (Hz): {formatValue(parameters.f0)}
                </Label>
                <Input
                  id="f0"
                  type="number"
                  value={parameters.f0}
                  onChange={(e) => updateParameter('f0', parseFloat(e.target.value) || 0)}
                  className="w-full"
                />
              </div>

              {/* 时间步长 */}
              <div className="space-y-3">
                <Label htmlFor="Ts" className="text-sm font-medium">
                  时间步长 (s): {parameters.Ts.toExponential(1)}
                </Label>
                <Input
                  id="Ts"
                  type="number"
                  step="1e-7"
                  value={parameters.Ts}
                  onChange={(e) => updateParameter('Ts', parseFloat(e.target.value) || 0)}
                  className="w-full"
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="mechanical" className="space-y-4">
            <div className="space-y-4">
              <div className="text-sm text-gray-600 mb-4">
                机械模式参数定义了超导腔的洛伦兹力失谐效应。每个模式包含谐振频率、品质因数和洛伦兹力系数。
              </div>
              
              {mechanicalModes.frequencies.map((freq, index) => (
                <Collapsible key={index}>
                  <CollapsibleTrigger className="flex items-center justify-between w-full p-3 bg-gray-50 rounded-lg hover:bg-gray-100">
                    <span className="font-medium">模式 {index + 1}: {freq} Hz</span>
                    <ChevronDown className="w-4 h-4" />
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pl-4">
                      {/* 频率 */}
                      <div className="space-y-2">
                        <Label className="text-sm">谐振频率 (Hz)</Label>
                        <Input
                          type="number"
                          value={mechanicalModes.frequencies[index]}
                          onChange={(e) => updateMechanicalMode(index, 'frequencies', parseFloat(e.target.value) || 0)}
                        />
                      </div>
                      
                      {/* 品质因数 */}
                      <div className="space-y-2">
                        <Label className="text-sm">品质因数 Q</Label>
                        <Input
                          type="number"
                          value={mechanicalModes.qualityFactors[index]}
                          onChange={(e) => updateMechanicalMode(index, 'qualityFactors', parseFloat(e.target.value) || 0)}
                        />
                      </div>
                      
                      {/* 洛伦兹力系数 */}
                      <div className="space-y-2">
                        <Label className="text-sm">洛伦兹力系数 K</Label>
                        <Input
                          type="number"
                          step="0.1"
                          value={mechanicalModes.lorentzCoefficients[index]}
                          onChange={(e) => updateMechanicalMode(index, 'lorentzCoefficients', parseFloat(e.target.value) || 0)}
                        />
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default ParameterControls;

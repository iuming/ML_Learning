import React, { useState, useEffect, useCallback } from 'react';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';
import ParameterControls from '@/components/ParameterControls';
import SimulationCharts from '@/components/SimulationCharts';
import { SRFPhysicsEngine, SRFParameters, MechanicalModes, SimulationResult } from '@/lib/srf-physics';

function App() {
  // 默认参数
  const [parameters, setParameters] = useState<SRFParameters>({
    Ts: 1e-6,
    f0: 1.3e9,
    QL: 3e6,
    roQ: 1036,
    beta: 1e4,
    ib: 0.008,
    dw0: 0,
    driveAmplitude: 1.0,
    simulationTime: 0.1
  });

  const [mechanicalModes, setMechanicalModes] = useState<MechanicalModes>({
    frequencies: [280, 341, 460, 487, 618],
    qualityFactors: [40, 20, 50, 80, 100],
    lorentzCoefficients: [2, 0.8, 2, 0.6, 0.2]
  });

  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [defaultConfig, setDefaultConfig] = useState<any>(null);

  // 加载配置文件
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch('/data/srf-config.json');
        const config = await response.json();
        setDefaultConfig(config);
        
        // 设置默认参数
        setParameters(config.defaultParameters);
        setMechanicalModes(config.mechanicalModes);
      } catch (error) {
        console.error('Failed to load configuration:', error);
        toast.error('无法加载配置文件');
      }
    };

    loadConfig();
  }, []);

  // 运行仿真
  const runSimulation = useCallback(async () => {
    setIsSimulating(true);
    toast.info('开始仿真计算...');
    
    try {
      // 使用 setTimeout 让UI有时间更新
      setTimeout(() => {
        try {
          console.log('Starting simulation with parameters:', parameters);
          const engine = new SRFPhysicsEngine(parameters, mechanicalModes);
          const result = engine.simulate();
          console.log('Simulation result:', result);
          setSimulationResult(result);
          toast.success('仿真计算完成！');
        } catch (error) {
          console.error('Simulation error:', error);
          toast.error('仿真计算出错，请检查参数设置');
        } finally {
          setIsSimulating(false);
        }
      }, 100);
    } catch (error) {
      setIsSimulating(false);
      console.error('Simulation error:', error);
      toast.error('仿真计算出错，请检查参数设置');
    }
  }, [parameters, mechanicalModes]);

  // 重置参数
  const resetParameters = useCallback(() => {
    if (defaultConfig) {
      setParameters(defaultConfig.defaultParameters);
      setMechanicalModes(defaultConfig.mechanicalModes);
      setSimulationResult(null);
      toast.info('参数已重置');
    }
  }, [defaultConfig]);

  // 参数变化时自动运行仿真
  useEffect(() => {
    if (defaultConfig && !isSimulating) {
      // 延迟执行，避免频繁计算
      const timer = setTimeout(() => {
        runSimulation();
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [parameters, mechanicalModes, defaultConfig, isSimulating, runSimulation]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* 页面标题 */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            虚拟超导射频腔仿真系统
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            基于洛伦兹力失谐效应的超导腔动态行为仿真。
            实时调整参数，观察电磁-机械耦合系统的复杂物理过程。
          </p>
        </div>

        {/* 主要内容区域 */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* 参数控制面板 */}
          <div className="xl:col-span-1">
            <ParameterControls
              parameters={parameters}
              mechanicalModes={mechanicalModes}
              onParametersChange={setParameters}
              onMechanicalModesChange={setMechanicalModes}
              onReset={resetParameters}
              onSimulate={runSimulation}
              isSimulating={isSimulating}
            />
          </div>

          {/* 图表显示区域 */}
          <div className="xl:col-span-2">
            <SimulationCharts
              data={simulationResult}
              isLoading={isSimulating}
            />
          </div>
        </div>

        {/* 技术说明 */}
        <div className="mt-12 bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">技术说明</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* 文字说明 */}
            <div className="space-y-6">
              <div className="space-y-2">
                <h3 className="font-medium text-gray-900">洛伦兹力失谐效应</h3>
                <p className="text-sm text-gray-600">
                  当腔内电磁场增强时，洛伦兹力会对腔壁施加压力，导致腔体机械形变，
                  进而改变谐振频率，形成电磁-机械耦合效应。这种效应是超导射频腔控制系统设计的核心挑战。
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-medium text-gray-900">多模式机械振动</h3>
                <p className="text-sm text-gray-600">
                  超导腔具有多个机械振动模式，每个模式都有特定的谐振频率和品质因数，
                  对腔体频率稳定性产生不同程度的影响。本仿真系统可以调整每个模式的参数。
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-medium text-gray-900">实时动态仿真</h3>
                <p className="text-sm text-gray-600">
                  基于状态空间模型的数值仿真，实时计算腔体电压、功率分布和机械振动，
                  为LLRF控制系统设计提供理论基础和性能预测。
                </p>
              </div>
            </div>
            
            {/* 图片展示 */}
            <div className="space-y-4">
              <div className="grid grid-cols-1 gap-4">
                <div className="text-center">
                  <img 
                    src="/images/srf-cavity.png" 
                    alt="超导射频腔结构" 
                    className="w-full h-48 object-cover rounded-lg shadow-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">超导射频腔结构示意图</p>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="text-center">
                    <img 
                      src="/images/em-field.png" 
                      alt="电磁场仿真" 
                      className="w-full h-24 object-cover rounded-lg shadow-sm"
                    />
                    <p className="text-xs text-gray-500 mt-1">电磁场分布</p>
                  </div>
                  <div className="text-center">
                    <img 
                      src="/images/vibration-chart.PNG" 
                      alt="振动频谱" 
                      className="w-full h-24 object-cover rounded-lg shadow-sm"
                    />
                    <p className="text-xs text-gray-500 mt-1">机械振动分析</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 页脚 */}
        <footer className="mt-12 text-center text-sm text-gray-500">
          <p>基于 LLRFLibsPy 库开发 | 超导射频腔物理仿真 | {new Date().getFullYear()}</p>
        </footer>
      </div>

      <Toaster />
    </div>
  );
}

export default App;

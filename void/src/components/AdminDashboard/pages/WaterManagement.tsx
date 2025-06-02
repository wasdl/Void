import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ComposedChart, Scatter } from 'recharts';
import {
  BeakerIcon, 
  ClockIcon, 
  UsersIcon, 
  PresentationChartLineIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

interface Cluster {
  id: string;
  users: number;
  accuracy: number;
  pattern: string;
  peakHours: string;
}

interface AccuracyComparison {
  cluster: string;
  personal: number;
  fd: number;
  average: number;
}

interface WaterSaving {
  mode: string;
  saving: number;
}

interface HourlySaving {
  hour: number;
  saving: number;
  usage: number;
}

interface MonthlyUsage {
  month: string;
  usage: number;
  baseline: number;
}

interface AnomalyStats {
  count: number;
  percentage: number;
  type: string;
}

interface PeakTimeAnalysis {
  timeSlot: string;
  usage: number;
  users: number;
}

interface PatternCumulative {
  time: string;
  A: number;
  B: number;
  C: number;
  D: number;
}

interface VersionStats {
  clusterCount: number;
  appliedCount: number;
  avgAccuracy: number;
  totalSaving: number;
  clusters: Cluster[];
  accuracyComparison: AccuracyComparison[];
  waterSaving: WaterSaving[];
  hourlyPatterns: {
    [key: string]: number[];
  };
  hourlySaving: HourlySaving[];
  monthlyUsage: {
    [key: string]: MonthlyUsage[];
  };
  anomalyStats: AnomalyStats[];
  peakTimeAnalysis: PeakTimeAnalysis[];
}

interface VersionData {
  [key: string]: VersionStats;
}

const accuracyComparison = [
  { cluster: 'Cluster A', personal: 92.72, fd: 76.61, average: 68.0032 },
  { cluster: 'Cluster B', personal: 89.7, fd: 68.44, average: 59.58 },
  { cluster: 'Cluster C', personal: 89.26, fd: 69.01, average: 67.32 },
  { cluster: 'Cluster D', personal: 89.84, fd: 67.51, average: 65.18 },
  { cluster: '평균', personal: 90.38, fd: 70.3925, average: 65.0208 },
];

// 그래프용 색상 팔레트 정의
const CHART_COLORS = [
  '#1428A0',
  '#00A9E0',
  '#00C853',
  '#EC4899',
];

const patternCumulativeData: PatternCumulative[] = [
  { time: '00:00', A: 99.8797, B: 100.219, C: 100.059, D: 99.9748 },
  { time: '00:30', A: 99.8917, B: 99.7599, C: 100.229, D: 99.7014 },
  { time: '01:00', A: 99.5863, B: 99.7107, C: 99.87, D: 100.022 },
  { time: '01:30', A: 100.394, B: 100.431, C: 99.8545, D: 99.5042 },
  { time: '02:00', A: 100.407, B: 99.8046, C: 100.205, D: 100.105 },
  { time: '02:30', A: 100.329, B: 99.6895, C: 100.07, D: 99.918 },
  { time: '03:00', A: 99.9001, B: 100.183, C: 100.003, D: 100.284 },
  { time: '03:30', A: 100.452, B: 99.9346, C: 99.4986, D: 100.113 },
  { time: '04:00', A: 99.8018, B: 100.156, C: 100.191, D: 99.8246 },
  { time: '04:30', A: 99.6934, B: 100.01, C: 99.7405, D: 100.243 },
  { time: '05:00', A: 100.28, B: 99.8796, C: 100.367, D: 128.069 },
  { time: '05:30', A: 99.9004, B: 99.8705, C: 100.432, D: 145.576 },
  { time: '06:00', A: 100.125, B: 100.155, C: 99.6858, D: 159.259 },
  { time: '06:30', A: 100.158, B: 100.472, C: 100.13, D: 165.907 },
  { time: '07:00', A: 100.697, B: 99.8878, C: 100.072, D: 173.414 },
  { time: '07:30', A: 99.848, B: 100.4, C: 100.427, D: 181.311 },
  { time: '08:00', A: 100.113, B: 100.011, C: 100.285, D: 179.618 },
  { time: '08:30', A: 99.5702, B: 100.085, C: 100.192, D: 179.019 },
  { time: '09:00', A: 100.225, B: 99.7373, C: 99.895, D: 175.914 },
  { time: '09:30', A: 100.321, B: 100.217, C: 100.319, D: 169.803 },
  { time: '10:00', A: 100.25, B: 131.581, C: 131.416, D: 162.136 },
  { time: '10:30', A: 100.069, B: 149.58, C: 150.024, D: 153.866 },
  { time: '11:00', A: 99.8913, B: 162.651, C: 160.947, D: 135.759 },
  { time: '11:30', A: 100.6, B: 171, C: 171.821, D: 119.82 },
  { time: '12:00', A: 100.795, B: 178.701, C: 176.878, D: 106.37 },
  { time: '12:30', A: 99.9485, B: 184.843, C: 185.928, D: 100.085 },
  { time: '13:00', A: 99.6697, B: 184.413, C: 183.073, D: 100.106 },
  { time: '13:30', A: 99.4703, B: 180.786, C: 183.348, D: 99.8323 },
  { time: '14:00', A: 100.086, B: 179.664, C: 178.443, D: 99.7084 },
  { time: '14:30', A: 100.059, B: 174.697, C: 174.677, D: 99.5285 },
  { time: '15:00', A: 100.186, B: 166.926, C: 164.678, D: 99.8746 },
  { time: '15:30', A: 100.166, B: 158.872, C: 156.333, D: 100.194 },
  { time: '16:00', A: 100.204, B: 139.398, C: 135.466, D: 99.8713 },
  { time: '16:30', A: 100.374, B: 122.525, C: 120.885, D: 99.67 },
  { time: '17:00', A: 137.355, B: 147.647, C: 107.823, D: 99.7606 },
  { time: '17:30', A: 157.77, B: 162.801, C: 100.45, D: 100.339 },
  { time: '18:00', A: 170.289, B: 175.556, C: 100.393, D: 99.9685 },
  { time: '18:30', A: 179.92, B: 182.968, C: 100.236, D: 99.8916 },
  { time: '19:00', A: 177.817, B: 183.208, C: 100.132, D: 100.374 },
  { time: '19:30', A: 178.703, B: 183.572, C: 100.166, D: 100.374 },
  { time: '20:00', A: 177.141, B: 181.883, C: 99.9605, D: 99.9838 },
  { time: '20:30', A: 176.477, B: 183.543, C: 99.8266, D: 100.063 },
  { time: '21:00', A: 171.892, B: 177.392, C: 99.4469, D: 99.9057 },
  { time: '21:30', A: 165.221, B: 170.228, C: 100.315, D: 99.6304 },
  { time: '22:00', A: 147.481, B: 152.127, C: 99.7244, D: 100.178 },
  { time: '22:30', A: 127.351, B: 133.348, C: 99.7991, D: 100.016 },
  { time: '23:00', A: 110.03, B: 112.415, C: 100.334, D: 99.9441 },
  { time: '23:30', A: 100.145, B: 100.381, C: 100.059, D: 99.9748 }
];

// versionStats를 타입 지정
const versionStats: VersionData = {
  'v1.0.0': {
    clusterCount: 2,
    appliedCount: 670,
    avgAccuracy: 91.21,
    totalSaving: 5240,
    clusters: [
      {
        id: 'Cluster A',
        users: 60,
        accuracy: 88,
        pattern: '저녁형 (18-23시 집중)', peakHours: '18-23시'
      },
      {
        id: 'Cluster B',
        users: 40,
        accuracy: 87,
        pattern: '일반형 (점심, 저녁 집중)', peakHours: '10-13시, 18-23시'
      }
    ],
    accuracyComparison: [
      { cluster: 'Cluster A', personal: 88, fd: 82, average: 78 },
      { cluster: 'Cluster B', personal: 87, fd: 81, average: 77 },
      { cluster: '평균', personal: 87.5, fd: 81.5, average: 77.5 }
    ],
    waterSaving: [
      { mode: '일반', saving: 821 },
      { mode: 'FD', saving: 2132 },
      { mode: '개인화', saving: 2441 }
    ],
    hourlyPatterns: {
      'Cluster A': [
        1000,0,0,0,1000,3000,8000,15000,20000,12000,8000,6000,5000,4000,5000,6000,7000,12000,15000,10000,7000,5000,3000,2000
      ],
      'Cluster B': [
        2000,1000,0,0,0,1000,2000,4000,6000,7000,8000,7000,6000,5000,6000,8000,12000,18000,22000,16000,12000,8000,5000,3000
      ]
    },
    hourlySaving: Array.from({ length: 24 }, (_, hour) => ({
      hour,
      saving: Math.round(1000 + Math.random() * 4000),
      usage: Math.round(5000 + Math.random() * 15000)
    })),
    monthlyUsage: {
      'Cluster A': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(80000 + Math.random() * 40000),
        baseline: 100000
      })),
      'Cluster B': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(70000 + Math.random() * 30000),
        baseline: 90000
      }))
    },
    anomalyStats: [
      { count: 156, percentage: 6.3, type: '과다 사용' },
      { count: 89, percentage: 3.6, type: '누수 의심' },
      { count: 45, percentage: 1.8, type: '패턴 이상' }
    ],
    peakTimeAnalysis: [
      { timeSlot: '아침 (6-9시)', usage: 18500, users: 1200 },
      { timeSlot: '점심 (11-14시)', usage: 12000, users: 800 },
      { timeSlot: '저녁 (17-20시)', usage: 20000, users: 1500 },
      { timeSlot: '심야 (22-5시)', usage: 5000, users: 300 }
    ]
  },
  'v2.0.0': {
    clusterCount: 3,
    appliedCount: 1021,
    avgAccuracy: 90.56,
    totalSaving: 8360,
    clusters: [
      {
        id: 'Cluster A',
        users: 40,
        accuracy: 90,
        pattern: '저녁형 (18-23시 집중)', peakHours: '18-23시'
      },
      {
        id: 'Cluster B',
        users: 35,
        accuracy: 89,
        pattern: '일반형 (점심, 저녁 집중)', peakHours: '10-13시, 18-23시'
      },
      {
        id: 'Cluster C',
        users: 25,
        accuracy: 88,
        pattern: '점심형 (10-13시 집중)', peakHours: '10-13시'
      }
    ],
    accuracyComparison: [
      { cluster: 'Cluster A', personal: 90, fd: 83, average: 79 },
      { cluster: 'Cluster B', personal: 89, fd: 82, average: 78 },
      { cluster: 'Cluster C', personal: 88, fd: 81, average: 77 },
      { cluster: '평균', personal: 89, fd: 82, average: 78 }
    ],
    waterSaving: [
      { mode: '일반', saving: 1181 },
      { mode: 'FD', saving: 3082 },
      { mode: '개인화', saving: 4061 }
    ],
    hourlyPatterns: {
      'Cluster A': [
        1000,0,0,0,1000,3000,8000,15000,20000,12000,8000,6000,5000,4000,5000,6000,7000,12000,15000,10000,7000,5000,3000,2000
      ],
      'Cluster B': [
        2000,1000,0,0,0,1000,2000,4000,6000,7000,8000,7000,6000,5000,6000,8000,12000,18000,22000,16000,12000,8000,5000,3000
      ],
      'Cluster C': [
        1000,0,0,0,0,1000,2000,3000,5000,12000,18000,20000,15000,12000,10000,8000,7000,6000,5000,4000,3000,2000,1000,1000
      ]
    },
    hourlySaving: Array.from({ length: 24 }, (_, hour) => ({
      hour,
      saving: Math.round(1000 + Math.random() * 4000),
      usage: Math.round(5000 + Math.random() * 15000)
    })),
    monthlyUsage: {
      'Cluster A': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(80000 + Math.random() * 40000),
        baseline: 100000
      })),
      'Cluster B': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(70000 + Math.random() * 30000),
        baseline: 90000
      })),
      'Cluster C': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(60000 + Math.random() * 20000),
        baseline: 80000
      }))
    },
    anomalyStats: [
      { count: 156, percentage: 6.3, type: '과다 사용' },
      { count: 89, percentage: 3.6, type: '누수 의심' },
      { count: 45, percentage: 1.8, type: '패턴 이상' }
    ],
    peakTimeAnalysis: [
      { timeSlot: '아침 (6-9시)', usage: 18500, users: 1200 },
      { timeSlot: '점심 (11-14시)', usage: 12000, users: 800 },
      { timeSlot: '저녁 (17-20시)', usage: 20000, users: 1500 },
      { timeSlot: '심야 (22-5시)', usage: 5000, users: 300 }
    ]
  },
  'v3.0.0': {
    clusterCount: 4,
    appliedCount: 1340,
    avgAccuracy: 90.38,
    totalSaving: 12345,
    clusters: [
      { id: 'Cluster A', users: 30, accuracy: 95, pattern: '저녁형 (18-23시 집중)', peakHours: '18-23시' },
      { id: 'Cluster B', users: 25, accuracy: 92, pattern: '일반형 (점심, 저녁 집중)', peakHours: '10-13시, 18-23시' },
      { id: 'Cluster C', users: 20, accuracy: 88, pattern: '점심형 (10-13시 집중)', peakHours: '10-13시' },
      { id: 'Cluster D', users: 25, accuracy: 91, pattern: '아침형 (6-9시 집중)', peakHours: '6-9시 집중' }
    ],
    accuracyComparison: accuracyComparison,
    waterSaving: [
      { mode: '일반', saving: 1116 },
      { mode: 'FD', saving: 2515 },
      { mode: '개인화', saving: 3928 }
    ],
    hourlyPatterns: {
      'Cluster A': [
        1000,0,0,0,1000,3000,8000,15000,20000,12000,8000,6000,5000,4000,5000,6000,7000,12000,15000,10000,7000,5000,3000,2000
      ],
      'Cluster B': [
        2000,1000,0,0,0,1000,2000,4000,6000,7000,8000,7000,6000,5000,6000,8000,12000,18000,22000,16000,12000,8000,5000,3000
      ],
      'Cluster C': [
        1000,0,0,0,0,1000,2000,3000,5000,12000,18000,20000,15000,12000,10000,8000,7000,6000,5000,4000,3000,2000,1000,1000
      ],
      'Cluster D': [
        3000,2000,1000,1000,2000,3000,5000,7000,8000,9000,8000,7000,8000,7000,8000,9000,8000,9000,8000,7000,6000,5000,4000,3000
      ]
    },
    hourlySaving: Array.from({ length: 24 }, (_, hour) => ({
      hour,
      saving: Math.round(1000 + Math.random() * 4000),
      usage: Math.round(5000 + Math.random() * 15000)
    })),
    monthlyUsage: {
      'Cluster A': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(80000 + Math.random() * 40000),
        baseline: 100000
      })),
      'Cluster B': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(70000 + Math.random() * 30000),
        baseline: 90000
      })),
      'Cluster C': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(60000 + Math.random() * 20000),
        baseline: 80000
      })),
      'Cluster D': Array.from({ length: 12 }, (_, i) => ({
        month: `${i + 1}월`,
        usage: Math.round(50000 + Math.random() * 10000),
        baseline: 70000
      }))
    },
    anomalyStats: [
      { count: 156, percentage: 6.3, type: '과다 사용' },
      { count: 89, percentage: 3.6, type: '누수 의심' },
      { count: 45, percentage: 1.8, type: '패턴 이상' }
    ],
    peakTimeAnalysis: [
      { timeSlot: '아침 (6-9시)', usage: 18500, users: 1200 },
      { timeSlot: '점심 (11-14시)', usage: 12000, users: 800 },
      { timeSlot: '저녁 (17-20시)', usage: 20000, users: 1500 },
      { timeSlot: '심야 (22-5시)', usage: 5000, users: 300 }
    ]
  }
};

const WaterManagement = () => {
  const [selectedVersion, setSelectedVersion] = useState('v3.0.0');
  const currentStats = versionStats[selectedVersion];

  const getFilteredPatternData = () => {
    if (selectedVersion === 'v1.0.0') {
      return patternCumulativeData.map(item => ({
        time: item.time,
        A: item.A,
        B: item.B
      }));
    } else if (selectedVersion === 'v2.0.0') {
      return patternCumulativeData.map(item => ({
        time: item.time,
        A: item.A,
        B: item.B,
        C: item.C
      }));
    } else {
      return patternCumulativeData;
    }
  };

  // 버전별 패턴 레이블 가져오기
  const getPatternLabels = () => {
    if (selectedVersion === 'v1.0.0') {
      return [
        { name: 'A 클러스터', color: CHART_COLORS[0] },
        { name: 'B 클러스터', color: CHART_COLORS[1] }
      ];
    } else if (selectedVersion === 'v2.0.0') {
      return [
        { name: 'A 클러스터', color: CHART_COLORS[0] },
        { name: 'B 클러스터', color: CHART_COLORS[1] },
        { name: 'C 클러스터', color: CHART_COLORS[2] }
      ];
    } else {
      return [
        { name: 'A 클러스터', color: CHART_COLORS[0] },
        { name: 'B 클러스터', color: CHART_COLORS[1] },
        { name: 'C 클러스터', color: CHART_COLORS[2] },
        { name: 'D 클러스터', color: CHART_COLORS[3] }
      ];
    }
  };

  return (
    <div className="space-y-2">
      <div className="bg-white rounded-lg border border-[#DFE3E7] p-1">
        <div className="flex items-center gap-1">
          {Object.keys(versionStats).map((version) => (
            <button
              key={version}
              onClick={() => setSelectedVersion(version)}
              className={`px-3 py-1.5 rounded-lg transition-all duration-200 text-sm font-medium ${
                selectedVersion === version
                  ? 'bg-[#1428A0] text-white'
                  : 'text-[#697077] hover:bg-[#F7F8FA]'
              }`}
            >
              {version}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2">
        {[
          { 
            title: '클러스터 수', 
            value: currentStats.clusterCount.toLocaleString(),
            trend: selectedVersion === 'v1.0.0' ? '' : '+1',
            trendColor: 'text-green-500',
            icon: <UsersIcon className="w-4 h-4" />
          },
          { 
            title: '적용 대수', 
            value: currentStats.appliedCount.toLocaleString(), 
            trend: selectedVersion === 'v1.0.0' ? '' : (selectedVersion === 'v2.0.0' ? '+30%' : '+35%'),
            trendColor: 'text-green-500',
            icon: <BeakerIcon className="w-4 h-4" />
          },
          { 
            title: '평균 정확도', 
            value: `${currentStats.avgAccuracy.toLocaleString()}%`, 
            trend:  selectedVersion === 'v1.0.0' ? '' : (selectedVersion === 'v2.0.0' ? '-0.18%':'-0.65%'),
            trendColor: 'text-red-500',
            icon: <PresentationChartLineIcon className="w-4 h-4" />
          },
          { 
            title: '총 절약량', 
            value: `${currentStats.totalSaving.toLocaleString()}L`,
            trend: selectedVersion === 'v1.0.0' ? '' : (selectedVersion === 'v2.0.0' ? '+45%':'+55%'),
            trendColor: 'text-green-500',
            icon: <BeakerIcon className="w-4 h-4" />
          }
        ].map((stat, index) => (
          <div 
            key={index} 
            className="bg-white rounded-lg p-2 border border-gray-200"
          >
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  <div className="p-1 rounded-lg bg-gray-50 text-gray-600">
                    {stat.icon}
                  </div>
                  <p className="text-xs font-normal text-gray-500">{stat.title}</p>
                </div>
                <div className={`flex items-center text-xs ${stat.trendColor}`}>
                  {stat.trend}
                </div>
              </div>
              <h3 className="text-base font-bold text-gray-900">{stat.value}</h3>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-9 gap-2">
        {/* 클러스터별 24시간 출수량 패턴 */}
        <div className="col-span-4 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-[#DFE3E7]">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-[#F7F8FA]">
                <ClockIcon className="w-4 h-4 text-[#1428A0]" />
              </div>
              <h3 className="text-sm font-bold text-[#1A1F27]">
                클러스터별 24시간 출수량 패턴
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px]">
              <div className="h-[300px]">
                <ResponsiveContainer>
                  <LineChart
                      data={patternCumulativeData}
                      margin={{top: 10, right: 20, left: 20, bottom: 10}}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2}/>
                    <XAxis
                        dataKey="time"
                        tick={{fontSize: 10, fill: '#697077'}}
                        interval={3}
                    />
                    <YAxis
                        domain={[95, 190]}
                        tick={{fontSize: 10, fill: '#697077'}}
                        ticks={[95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]}
                        tickFormatter={(value) => value.toFixed(2)}
                        label={{
                          value: '출수량 (ml)',
                          angle: -90,
                          position: 'insideLeft',
                          offset: 5,
                          style: {fontSize: 11, fill: '#697077'}
                        }}
                    />
                    <Tooltip
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #DFE3E7',
                          borderRadius: '8px',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                        }}
                        formatter={(value: number) => [`${value.toFixed(2)}ml`, '출수량']}
                    />
                    <Line
                        type="monotone"
                        dataKey="A"
                        stroke={CHART_COLORS[0]}
                        strokeWidth={2}
                        dot={{r: 2}}
                        activeDot={{r: 4}}
                        connectNulls={true}
                    />
                    <Line
                        type="monotone"
                        dataKey="B"
                        stroke={CHART_COLORS[1]}
                        strokeWidth={2}
                        dot={{r: 2}}
                        activeDot={{r: 4}}
                        connectNulls={true}
                    />
                    {/* 조건부 라인 */}
                    {/* C는 공통이거나 v2.0.0 전용 */}
                    {(selectedVersion === 'v2.0.0' || selectedVersion === 'v3.0.0') && (
                        <Line
                            type="monotone"
                            dataKey="C"
                            stroke={CHART_COLORS[2]}
                            strokeWidth={2}
                            dot={{r: 2}}
                            activeDot={{r: 4}}
                            connectNulls={true}
                        />
                    )}
                    {selectedVersion === 'v3.0.0' && (
                        <Line
                            type="monotone"
                            dataKey="D"
                            stroke={CHART_COLORS[3]}
                            strokeWidth={2}
                            dot={{r: 2}}
                            activeDot={{r: 4}}
                            connectNulls={true}
                        />

                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-center">
                {(
                  selectedVersion === 'v1.0.0'
                    ? [
                      {name: 'A 클러스터', color: CHART_COLORS[0]},
                      {name: 'B 클러스터', color: CHART_COLORS[1]}
                    ]
                    : selectedVersion === 'v2.0.0'
                      ? [
                        {name: 'A 클러스터', color: CHART_COLORS[0]},
                        {name: 'B 클러스터', color: CHART_COLORS[1]},
                        {name: 'C 클러스터', color: CHART_COLORS[2]}
                      ]
                      : [
                        {name: 'A 클러스터', color: CHART_COLORS[0]},
                        {name: 'B 클러스터', color: CHART_COLORS[1]},
                        {name: 'C 클러스터', color: CHART_COLORS[2]},
                        {name: 'D 클러스터', color: CHART_COLORS[3]}
                      ]
                ).map((item) => (
                  <div
                    key={item.name}
                    className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full"
                  >
                    <div
                      className="w-1.5 h-1.5 rounded-full"
                      style={{backgroundColor: item.color}}
                    />
                    <span className="text-xs text-gray-600 whitespace-nowrap">
                      {item.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 모드별 물 절약량 */}
        <div className="col-span-3 bg-white rounded-lg border border-[#DFE3E7] ">
          <div className="bg-white rounded-lg ">
            <div className="p-2 border-b border-gray-100">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-gray-50">
                  <BeakerIcon className="w-4 h-4 text-gray-600"/>
                </div>
                <h3 className="text-sm font-bold text-gray-900 tracking-tight">
                  모드별 물 절약량
                </h3>
              </div>
            </div>
            <div className="p-2 mt-12">
              <div className="min-h-[300px] flex flex-col">
                <div className="h-[250px]">
                  <ResponsiveContainer>
                    <BarChart
                        data={currentStats.waterSaving}
                        margin={{top: 10, right: 20, left: 20, bottom: 10}}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis 
                        dataKey="mode" 
                        tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      />
                      <YAxis 
                        tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                        label={{ 
                          value: '절약량 (L)', 
                          angle: -90, 
                          position: 'insideLeft',
                          offset: 20,
                          style: { fontSize: 11, fontFamily: 'SamsungOne' }
                        }}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: 'none',
                          borderRadius: '8px',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                          padding: '6px 10px',
                          fontSize: '11px',
                          fontFamily: 'SamsungOne'
                        }}
                        formatter={(value: number) => [`${value.toLocaleString()}L`, '절약량']}
                      />
                      <Bar 
                        dataKey="saving" 
                        name="절약량" 
                        radius={[4, 4, 0, 0]}
                      >
                        {currentStats.waterSaving.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={CHART_COLORS[index]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-1 flex flex-wrap gap-1 justify-center">
                  {currentStats.waterSaving.map((item, index) => (
                    <div 
                      key={item.mode}
                      className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full"
                    >
                      <div 
                        className="w-1.5 h-1.5 rounded-full" 
                        style={{ backgroundColor: CHART_COLORS[index] }}
                      />
                      <span className="text-xs text-gray-600 whitespace-nowrap">
                        {item.mode}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 군집별 유저 비율 */}
        <div className="col-span-2 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-[#DFE3E7]">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-[#F7F8FA]">
                <UsersIcon className="w-4 h-4 text-[#1428A0]" />
              </div>
              <h3 className="text-sm font-bold text-[#1A1F27]">
                군집별 유저 비율
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px] flex flex-col">
              <div className="h-[250px]">
                <ResponsiveContainer>
                  <PieChart margin={{ top: 10, right: 20, left: 35, bottom: 10 }}>
                    <Pie
                      data={currentStats.clusters}
                      dataKey="users"
                      nameKey="id"
                      cx="50%"
                      cy="45%"
                      innerRadius={50}
                      outerRadius={70}
                      label={({ percent }) => `${(percent * 100).toFixed(1)}%`}
                      labelLine={true}
                      style={{ fontFamily: 'SamsungOne' }}
                    >
                      {currentStats.clusters.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={CHART_COLORS[index]}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontFamily: 'SamsungOne'
                      }}
                      formatter={(value: number, name: string) => [
                        `${value.toLocaleString()}명`,
                        currentStats.clusters.find(c => c.id === name)?.pattern || name
                      ]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-left">
                {currentStats.clusters.map((cluster, index) => (
                  <div
                    key={cluster.id}
                    className="flex items-center gap-1 px-2 py-0.5 bg-[#F7F8FA] rounded-lg text-xs text-[#697077]"
                  >
                    <div
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ backgroundColor: CHART_COLORS[index] }}
                    />
                    <span>{cluster.id} ({cluster.pattern})</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>


      </div>

      <div className="grid grid-cols-9 gap-2">
        {/* 군집별 정확도 비교 */}
        <div className="col-span-4 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-gray-50">
                <PresentationChartLineIcon className="w-4 h-4 text-gray-600" />
              </div>
              <h3 className="text-sm font-bold text-gray-900 tracking-tight">
                군집별 정확도 비교
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px] flex flex-col">
              <div className="h-[250px]">
                <ResponsiveContainer>
                  <LineChart 
                    data={currentStats.accuracyComparison}
                    margin={{ top: 10, right: 20, left: 20, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis 
                      dataKey="cluster" 
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      padding={{ left: 20, right: 20 }}
                    />
                    <YAxis 
                      domain={[70, 100]}
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      ticks={[55,60,65,70, 75, 80, 85, 90, 95, 100]}
                      label={{ 
                        value: '정확도 (%)', 
                        angle: -90, 
                        position: 'insideLeft',
                        offset: 20,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontFamily: 'SamsungOne'
                      }}
                      formatter={(value: number) => [`${value.toFixed(1)}%`, '정확도']}
                    />
                    <Line type="monotone" dataKey="personal" stroke={CHART_COLORS[0]} name="개인화 모델" strokeWidth={2} dot={{ strokeWidth: 2, r: 3 }} />
                    <Line type="monotone" dataKey="fd" stroke={CHART_COLORS[1]} name="FD 모델" strokeWidth={2} dot={{ strokeWidth: 2, r: 3 }} />
                    <Line type="monotone" dataKey="average" stroke={CHART_COLORS[2]} name="평균" strokeWidth={2} dot={{ strokeWidth: 2, r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-center">
                {[
                  { name: '개인화 모델', color: CHART_COLORS[0] },
                  { name: '분류 후 모델', color: CHART_COLORS[1] },
                  { name: '전체 모델', color: CHART_COLORS[2] }
                ].map((item) => (
                  <div 
                    key={item.name}
                    className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full"
                  >
                    <div 
                      className="w-1.5 h-1.5 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-xs text-gray-600 whitespace-nowrap">
                      {item.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 시간대별 평균 절수율 */}
        <div className="col-span-5 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-gray-50">
                <BeakerIcon className="w-4 h-4 text-gray-600" />
              </div>
              <h3 className="text-sm font-bold text-gray-900 tracking-tight">
                시간대별 평균 절수율
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px] flex flex-col">
              <div className="h-[250px]">
                <ResponsiveContainer>
                  <AreaChart
                    data={currentStats.hourlySaving}
                    margin={{ top: 10, right: 20, left: 20, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis
                      dataKey="hour"
                      ticks={[0,3,6,9,12,15,18,21,23]}
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      label={{
                        value: '시간',
                        position: 'bottom',
                        offset: -10,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <YAxis
                      yAxisId="left"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      label={{
                        value: '절수량 (L)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 20,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      label={{
                        value: '사용량 (ml)',
                        angle: 90,
                        position: 'insideRight',
                        offset: -25,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontFamily: 'SamsungOne'
                      }}
                      formatter={(value: number, name: string) => {
                        if (name === 'saving') return [`${value.toLocaleString()}L`, '절수량'];
                        return [`${value.toLocaleString()}ml`, '사용량'];
                      }}
                      labelFormatter={(hour: number) => `${hour}시`}
                    />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="saving"
                      stroke={CHART_COLORS[0]}
                      fill={`${CHART_COLORS[0]}33`}
                      name="절수량"
                      strokeWidth={2}
                    />
                    <Area
                      yAxisId="right"
                      type="monotone"
                      dataKey="usage"
                      stroke={CHART_COLORS[1]}
                      fill={`${CHART_COLORS[1]}33`}
                      name="사용량"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-center">
                {['절수량', '사용량'].map((name, index) => (
                  <div 
                    key={name}
                    className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full"
                  >
                    <div 
                      className="w-1.5 h-1.5 rounded-full" 
                      style={{ backgroundColor: CHART_COLORS[index] }}
                    />
                    <span className="text-xs text-gray-600 whitespace-nowrap">
                      {name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-9 gap-2">
        {/* 이상 사용량 통계 */}
        <div className="col-span-3 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-gray-50">
                <ExclamationTriangleIcon className="w-4 h-4 text-gray-600" />
              </div>
              <h3 className="text-sm font-bold text-gray-900 tracking-tight">
                이상 사용량 통계
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px] flex flex-col">
              <div className="h-[250px]">
                <ResponsiveContainer>
                  <RadarChart
                    data={currentStats.anomalyStats}
                    margin={{ top: 10, right: 20, left: 20, bottom: 10 }}
                  >
                    <PolarGrid stroke="#e5e7eb" />
                    <PolarAngleAxis
                      dataKey="type"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                    />
                    <PolarRadiusAxis
                      angle={30}
                      domain={[0, 200]}
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                    />
                    <Radar
                      name="건수"
                      dataKey="count"
                      stroke={CHART_COLORS[0]}
                      fill={`${CHART_COLORS[0]}33`}
                      fillOpacity={0.6}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontFamily: 'SamsungOne'
                      }}
                      formatter={(value: number) => [`${value.toLocaleString()}건`, '건수']}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-center">
                <div className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full">
                  <div 
                    className="w-1.5 h-1.5 rounded-full" 
                    style={{ backgroundColor: CHART_COLORS[0] }}
                  />
                  <span className="text-xs text-gray-600 whitespace-nowrap">
                    이상 사용량 건수
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 클러스터별 월간 사용량 추이 */}
        <div className="col-span-6 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-gray-50">
                <PresentationChartLineIcon className="w-4 h-4 text-gray-600" />
              </div>
              <h3 className="text-sm font-bold text-gray-900 tracking-tight">
                클러스터별 월간 사용량 추이
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px] flex flex-col">
              <div className="h-[250px]">
                <ResponsiveContainer>
                  <ComposedChart
                    data={Object.values(currentStats.monthlyUsage)[0]}
                    margin={{ top: 10, right: 20, left: 20, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis
                      dataKey="month"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      label={{
                        value: '사용량 (L)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: -20,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontFamily: 'SamsungOne'
                      }}
                      formatter={(value: number, name: string) => {
                        if (name === 'baseline') return [`${value.toLocaleString()}L`, '기준값'];
                        return [`${value.toLocaleString()}L`, '사용량'];
                      }}
                    />
                    {Object.entries(currentStats.monthlyUsage).map(([clusterId, data], index) => (
                      <React.Fragment key={clusterId}>
                        <Bar
                          dataKey="usage"
                          name={clusterId}
                          fill={CHART_COLORS[index]}
                          radius={[4, 4, 0, 0]}
                          stackId="stack"
                        />
                        <Scatter
                          name={`${clusterId} 기준값`}
                          data={data}
                          fill={CHART_COLORS[index]}
                          line={{ stroke: CHART_COLORS[index] }}
                          dataKey="baseline"
                        />
                      </React.Fragment>
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-center">
                {Object.keys(currentStats.monthlyUsage).map((clusterId, index) => (
                  <div 
                    key={clusterId}
                    className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full"
                  >
                    <div 
                      className="w-1.5 h-1.5 rounded-full" 
                      style={{ backgroundColor: CHART_COLORS[index] }}
                    />
                    <span className="text-xs text-gray-600 whitespace-nowrap">
                      {clusterId} ({currentStats.clusters.find(c => c.id === clusterId)?.pattern})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-9 gap-2">
        {/* 시간대별 사용량 및 사용자 수 */}
        <div className="col-span-9 bg-white rounded-lg border border-[#DFE3E7]">
          <div className="p-2 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-gray-50">
                <ClockIcon className="w-4 h-4 text-gray-600" />
              </div>
              <h3 className="text-sm font-bold text-gray-900 tracking-tight">
                시간대별 사용량 및 사용자 수
              </h3>
            </div>
          </div>
          <div className="p-2">
            <div className="min-h-[300px] flex flex-col">
              <div className="h-[250px]">
                <ResponsiveContainer>
                  <BarChart
                    data={currentStats.peakTimeAnalysis}
                    margin={{ top: 10, right: 20, left: 20, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis
                      dataKey="timeSlot"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      interval={0}
                      angle={0}
                      textAnchor="middle"
                      height={50}
                    />
                    <YAxis
                      yAxisId="left"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      label={{
                        value: '사용량 (L)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 10,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={{ fontSize: 10, fontFamily: 'SamsungOne' }}
                      label={{
                        value: '사용자 수',
                        angle: 90,
                        position: 'insideRight',
                        offset: -20,
                        style: { fontSize: 11, fontFamily: 'SamsungOne' }
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontFamily: 'SamsungOne'
                      }}
                      formatter={(value: number, name: string) => {
                        if (name === 'usage') return [`${value.toLocaleString()}L`, '사용량'];
                        return [`${value.toLocaleString()}명`, '사용자 수'];
                      }}
                    />
                    <Bar yAxisId="left" dataKey="usage" name="사용량" fill={CHART_COLORS[0]} radius={[4, 4, 0, 0]} />
                    <Line yAxisId="right" type="monotone" dataKey="users" name="사용자 수" stroke={CHART_COLORS[1]} strokeWidth={2} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-1 justify-center">
                {['사용량', '사용자 수'].map((name, index) => (
                  <div 
                    key={name}
                    className="flex items-center gap-1 px-2 py-0.5 bg-gray-50 rounded-full"
                  >
                    <div 
                      className="w-1.5 h-1.5 rounded-full" 
                      style={{ backgroundColor: CHART_COLORS[index] }}
                    />
                    <span className="text-xs text-gray-600 whitespace-nowrap">
                      {name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WaterManagement; 
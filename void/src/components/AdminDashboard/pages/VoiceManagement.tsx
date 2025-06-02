import React, { useState, useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { 
  voiceSamples, 
  genderData,
  familyGroups,
} from '../data/voiceSampleData';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Legend } from 'recharts';

// 클러스터 데이터 포인트 타입 정의
interface ClusterDataPoint {
  x: number;
  y: number;
  id: string;
  gender: '남성' | '여성';
  age: number;
  accuracy: number;
  isSubPoint?: boolean;
  fillOpacity?: number;
  radius?: number;
}

// K-means 클러스터링 결과를 시각화하는 컴포넌트
const ClusteringVisualization = ({ familyId }: { familyId: string }) => {
  // 해당 가족의 음성 데이터 특성을 2D 공간에 매핑
  // 실제로는 고차원 음성 특성이 2D로 축소된 값이 들어가야 합니다
  const getClusterData = (familyId: string): ClusterDataPoint[] => {
    const familyVoices = voiceSamples.filter(sample => sample.familyId === familyId);
    const result: ClusterDataPoint[] = [];
    
    // 각 음성 샘플에 대해 여러 데이터 포인트 생성
    familyVoices.forEach(voice => {
      // 기본 데이터 포인트
      const baseX = voice.accuracy * Math.cos(voice.age / 62);
      const baseY = voice.accuracy * Math.sin(voice.age / 22);
      
      // 기본 데이터 포인트 추가
      result.push({
        x: Number(baseX.toFixed(2)),
        y: Number(baseY.toFixed(2)),
        id: voice.id,
        gender: voice.gender,
        age: voice.age,
        accuracy: voice.accuracy
      });

      // 각 음성 샘플 주변에 추가 데이터 포인트 생성 (클러스터 형성)
      for (let i = 0; i < 5; i++) {
        const variance = 3; // 분산 정도
        const randomX = baseX + (Math.random() - 0.5) * variance;
        const randomY = baseY + (Math.random() - 0.5) * variance;
        
        result.push({
          x: Number(randomX.toFixed(2)),
          y: Number(randomY.toFixed(2)),
          id: `${voice.id}_${i + 1}`,
          gender: voice.gender,
          age: voice.age,
          accuracy: voice.accuracy,
          isSubPoint: true // 부가 데이터 포인트 표시
        });
      }
    });
    
    return result;
  };

  const clusterData = getClusterData(familyId);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-[#DFE3E7] p-6">
      <h4 className="text-sm font-medium text-[#1A1F27] mb-4">
        {familyId} 음성 클러스터 분석
      </h4>
      <div className="h-[300px]">
        <ScatterChart
          width={500}
          height={300}
          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
        >
          <XAxis 
            type="number" 
            dataKey="x" 
            name="특성 1" 
            unit=""
            domain={['auto', 'auto']}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="특성 2" 
            unit=""
            domain={['auto', 'auto']}
          />
          <ZAxis 
            type="number" 
            range={[100, 200]} 
            domain={['auto', 'auto']}
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (payload && payload.length > 0) {
                const data = payload[0].payload;
                return (
                  <div className="bg-white p-2 shadow-lg rounded-lg border border-[#DFE3E7]">
                    <p className="font-medium text-[#1A1F27]">{data.id}</p>
                    <p className="text-sm text-[#697077]">{data.gender} • {data.age}세</p>
                    <p className="text-sm text-[#697077]">정확도: {data.accuracy}%</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend />
          <Scatter
            name="남성"
            data={clusterData.filter(d => d.gender === '남성').map(d => ({
              ...d,
              fillOpacity: d.isSubPoint ? 0.3 : 0.8,
              radius: d.isSubPoint ? 4 : 6
            }))}
            fill="#1428A0"
            shape="circle"
          />
          <Scatter
            name="여성"
            data={clusterData.filter(d => d.gender === '여성').map(d => ({
              ...d,
              fillOpacity: d.isSubPoint ? 0.3 : 0.8,
              radius: d.isSubPoint ? 4 : 6
            }))}
            fill="#697077"
            shape="circle"
          />
        </ScatterChart>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        {clusterData.map(data => (
          <div 
            key={data.id}
            className="flex items-center justify-between p-2 bg-[#F7F8FA] rounded-lg"
          >
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full
                ${data.gender === '남성' ? 'bg-[#1428A0]' : 'bg-[#697077]'}`} 
              />
              <span className="text-sm font-medium text-[#1A1F27]">{data.id}</span>
            </div>
            <span className="text-sm text-[#697077]">
              {data.gender} • {data.age}세
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const VoiceManagement = () => {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [accuracyFilter, setAccuracyFilter] = useState('전체');
  const [genderFilter, setGenderFilter] = useState('전체');
  const [ageFilter, setAgeFilter] = useState('전체');

  // 필터링된 데이터
  const filteredSamples = useMemo(() => {
    return voiceSamples.filter(sample => {
      // 검색어 필터링 (VOICE ID나 HOME ID로 검색)
      const searchMatch = 
        sample.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        sample.familyId.toLowerCase().includes(searchTerm.toLowerCase());
      
      // 정확도 필터링
      let accuracyMatch = true;
      if (accuracyFilter === '90% 이상') accuracyMatch = sample.accuracy >= 90;
      else if (accuracyFilter === '80-90%') accuracyMatch = sample.accuracy >= 80 && sample.accuracy < 90;
      else if (accuracyFilter === '80% 미만') accuracyMatch = sample.accuracy < 80;

      // 성별 필터링
      const genderMatch = genderFilter === '전체' || sample.gender === genderFilter;

      // 연령대 필터링
      let ageMatch = true;
      if (ageFilter === '10대 이하') ageMatch = sample.age < 20;
      else if (ageFilter === '20-30대') ageMatch = sample.age >= 20 && sample.age < 40;
      else if (ageFilter === '40-50대') ageMatch = sample.age >= 40 && sample.age < 60;
      else if (ageFilter === '60대 이상') ageMatch = sample.age >= 60;

      return searchMatch && accuracyMatch && genderMatch && ageMatch;
    });
  }, [searchTerm, accuracyFilter, genderFilter, ageFilter]);

  // 차트 커스텀 툴팁
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 shadow-lg rounded-lg border border-gray-100">
          <p className="font-medium">{payload[0].name}</p>
          <p className="text-gray-600">
            {payload[0].value}명 ({((payload[0].value / 110) * 100).toFixed(1)}%)
          </p>
        </div>
      );
    }
    return null;
  };

  // 현재 표시된 가족들의 고유 ID 추출
  const uniqueFamilyIds = filteredSamples
    .map(sample => sample.familyId)
    .filter((familyId, index, self) => 
      self.indexOf(familyId) === index
    );

  return (
    <div className="flex gap-4">
      {/* 왼쪽 패널 - 고정 높이로 변경 */}
      <div className="w-2/3">
        <div className="space-y-4">
          {/* 통계 카드 그리드 - 고정 */}
          <div className="grid grid-cols-4 gap-3">
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-3">
              <p className="text-xs font-medium text-gray-500">전체 음성 샘플</p>
              <div className="flex items-baseline mt-1">
                <h3 className="text-lg font-bold text-gray-900">2,458</h3>
                <span className="ml-1 text-xs text-gray-500">개</span>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-3">
              <p className="text-xs font-medium text-gray-500">평균 정확도</p>
              <div className="flex items-baseline mt-1">
                <h3 className="text-lg font-bold text-gray-900">74.8</h3>
                <span className="ml-1 text-xs text-gray-500">%</span>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-3">
              <p className="text-xs font-medium text-gray-500">신규 등록</p>
              <div className="flex items-baseline mt-1">
                <h3 className="text-lg font-bold text-gray-900">127</h3>
                <span className="ml-1 text-xs text-gray-500">개</span>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-3">
              <p className="text-xs font-medium text-gray-500">활성 HOME</p>
              <div className="flex items-baseline mt-1">
                <h3 className="text-lg font-bold text-gray-900">89</h3>
                <span className="ml-1 text-xs text-gray-500">개</span>
              </div>
            </div>
          </div>

          {/* 필터와 테이블 - 고정 높이 */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-100">
            <div className="p-3 border-b border-gray-100">
              <div className="flex flex-wrap gap-2">
                <input 
                  type="text" 
                  placeholder="VOICE ID / HOME ID 검색" 
                  className="px-2 py-1 text-sm border border-gray-200 rounded-lg flex-1"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                <select 
                  className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg"
                  value={accuracyFilter}
                  onChange={(e) => setAccuracyFilter(e.target.value)}
                >
                  <option>전체</option>
                  <option>90% 이상</option>
                  <option>80-90%</option>
                  <option>80% 미만</option>
                </select>
                <select 
                  className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg"
                  value={genderFilter}
                  onChange={(e) => setGenderFilter(e.target.value)}
                >
                  <option>전체</option>
                  <option>남성</option>
                  <option>여성</option>
                </select>
                <select 
                  className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg"
                  value={ageFilter}
                  onChange={(e) => setAgeFilter(e.target.value)}
                >
                  <option>전체</option>
                  <option>10대 이하</option>
                  <option>20-30대</option>
                  <option>40-50대</option>
                  <option>60대 이상</option>
                </select>
              </div>
            </div>
            <div className="h-[400px] overflow-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600"></th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">HOME ID</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">VOICE ID</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">등록일</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">정확도</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">관리</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {filteredSamples.map((sample) => (
                    <React.Fragment key={sample.id}>
                      <tr className="hover:bg-gray-50">
                        <td className="px-4 py-4">
                          <button 
                            onClick={() => setExpandedRow(expandedRow === sample.id ? null : sample.id)}
                            className="text-gray-500 hover:text-gray-700"
                          >
                            {expandedRow === sample.id ? '▼' : '▶'}
                          </button>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex flex-col">
                            <span className="text-sm font-medium text-gray-900">
                              {sample.familyId}
                            </span>
                            <span className="text-xs text-gray-500">
                              구성원: {familyGroups.find(f => f.familyId === sample.familyId)?.members}명
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex flex-col">
                            <span className="text-sm font-medium text-gray-900">
                              {sample.id}
                            </span>
                            <span className="text-xs text-gray-500">
                              {sample.gender} • {sample.age}세
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-600">
                          {sample.registeredAt}
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-blue-600 rounded-full"
                                style={{ width: `${sample.accuracy}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium">{sample.accuracy}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm space-x-2">
                          <button className="px-2 py-1 text-blue-600 hover:text-blue-800 
                                            border border-blue-600 rounded-lg">
                            재생
                          </button>
                          <button className="px-2 py-1 text-red-600 hover:text-red-800 
                                            border border-red-600 rounded-lg">
                            삭제
                          </button>
                        </td>
                      </tr>
                      {expandedRow === sample.id && (
                        <tr>
                          <td colSpan={6}>
                            <div className="px-4 py-3 bg-gray-50">
                              <h4 className="text-sm font-medium text-gray-700 mb-2">
                                {sample.id} 사용 로그
                              </h4>
                              <div className="space-y-2">
                                {[...sample.logs]
                                    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                                    .map((log, index) => (
                                  <div 
                                    key={index}
                                    className="space-y-2 bg-white p-3 rounded-lg border border-gray-200"
                                  >
                                    <div className="flex items-center justify-between">
                                      <div className="flex items-center gap-4">
                                        <span className="text-gray-500">{log.timestamp}</span>
                                        <span>{log.action}</span>
                                      </div>
                                      <span className={`px-2 py-1 rounded-full text-xs
                                        ${log.result === '성공' 
                                          ? 'bg-green-50 text-green-600' 
                                          : 'bg-red-50 text-red-600'}`}>
                                        {log.result}
                                      </span>
                                    </div>
                                    {log.clusteringResults && (
                                      <div className="mt-2 space-y-1">
                                        <p className="text-xs text-gray-500 font-medium">클러스터링 결과:</p>
                                        <div className="grid grid-cols-2 gap-2">
                                          {log.clusteringResults.map((result, i) => (
                                            <div key={i} className="flex items-center gap-2 text-sm">
                                              <span className="font-medium">{result.voiceId}:</span>
                                              <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                                <div 
                                                  className={`h-full rounded-full ${
                                                    result.similarity >= 85 
                                                      ? 'bg-green-500' 
                                                      : result.similarity >= 70 
                                                        ? 'bg-yellow-500' 
                                                        : 'bg-gray-400'
                                                  }`}
                                                  style={{ width: `${result.similarity}%` }}
                                                />
                                              </div>
                                              <span className="text-xs text-gray-600">
                                                {result.similarity}%
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 클러스터 분석 섹션 */}
          <div>
            <h3 className="text-sm font-bold text-gray-800 mb-4">음성 클러스터 분석</h3>
            <div className="h-[700px] overflow-auto pr-2">
              <div className="space-y-4">
                {uniqueFamilyIds.map(familyId => (
                  <ClusteringVisualization key={familyId} familyId={familyId} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 오른쪽 패널 - 고정 높이로 변경 */}
      <div className="w-1/3">
        <div className="space-y-4">
          {/* 성별 분포 차트 */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
            <div className="flex items-center justify-between mb-4">
              <p className="text-base font-medium text-gray-700">성별 분포</p>
              <div className="flex gap-6 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                  <span className="font-medium">남성 51%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-pink-500 rounded-full"></div>
                  <span className="font-medium">여성 49%</span>
                </div>
              </div>
            </div>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={genderData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {genderData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.color} 
                        stroke="none"
                      />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 실시간 인증 현황 */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-3">
            <h3 className="text-xs font-medium text-gray-500 mb-3">실시간 인증 현황</h3>
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg text-xs">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${i % 2 === 0 ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className="font-medium">VOICE_{String(i+1).padStart(3, '0')}</span>
                  </div>
                  <span className="text-gray-500">방금 전</span>
                </div>
              ))}
            </div>
          </div>

          {/* 연령대별 분포 */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-3">
            <h3 className="text-xs font-medium text-gray-500 mb-3">연령대별 분포</h3>
            <div className="space-y-2">
              {[
                { age: '10대 이하', percent: 15 },
                { age: '20-30대', percent: 35 },
                { age: '40-50대', percent: 40 },
                { age: '60대 이상', percent: 10 }
              ].map((data) => (
                <div key={data.age} className="flex items-center gap-2">
                  <span className="text-xs w-16">{data.age}</span>
                  <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${data.percent}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500">{data.percent}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VoiceManagement; 
import React from 'react';

const Dashboard = () => {
  return (
    <div>
      {/* 통계 카드 그리드 */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {[
          { title: '전체 사용자 수', value: '1,234', unit: '명' },
          { title: '전체 정수기 수', value: '5,678', unit: '대' },
          { title: '일일 평균 출수량', value: '9,876', unit: 'ml' }
        ].map((item, index) => (
          <div key={index} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <p className="text-sm font-medium text-gray-500 mb-2">{item.title}</p>
            <div className="flex items-baseline">
              <h3 className="text-2xl font-bold text-gray-900">{item.value}</h3>
              <span className="ml-1 text-gray-500">{item.unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* 차트 영역 */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 col-span-2">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-gray-800">전체 출수량 추이</h3>
            <select className="px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-lg text-sm">
              <option>최근 7일</option>
              <option>최근 30일</option>
              <option>최근 90일</option>
            </select>
          </div>
          <div className="h-[400px] bg-gray-50 rounded-lg" />
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 
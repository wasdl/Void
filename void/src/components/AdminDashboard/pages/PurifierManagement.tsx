import React from 'react';

const PurifierManagement = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h3 className="font-bold text-gray-800 mb-4">정수기 상태 현황</h3>
        <div className="grid grid-cols-4 gap-4">
          {[
            { title: '전체 정수기', value: '5,678', color: 'bg-gray-100' },
            { title: '정상 작동', value: '5,234', color: 'bg-green-100' },
            { title: '점검 필요', value: '342', color: 'bg-yellow-100' },
            { title: '고장', value: '102', color: 'bg-red-100' }
          ].map((item, index) => (
            <div key={index} className={`${item.color} rounded-lg p-4`}>
              <p className="text-sm font-medium text-gray-600">{item.title}</p>
              <p className="text-2xl font-bold mt-2">{item.value}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="font-bold text-gray-800">정수기 목록</h3>
          <div className="flex gap-4">
            <input 
              type="text" 
              placeholder="정수기 ID 검색" 
              className="px-4 py-2 border border-gray-200 rounded-lg"
            />
            <select className="px-4 py-2 border border-gray-200 rounded-lg">
              <option>전체 상태</option>
              <option>정상 작동</option>
              <option>점검 필요</option>
              <option>고장</option>
            </select>
          </div>
        </div>

        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">정수기 ID</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">설치 위치</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">상태</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">마지막 점검일</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">관리</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {/* 샘플 데이터 */}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PurifierManagement; 
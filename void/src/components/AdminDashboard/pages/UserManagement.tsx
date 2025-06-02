import React from 'react';

const UserManagement = () => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex justify-between items-center mb-6">
        <div className="flex gap-4">
          <input 
            type="text" 
            placeholder="사용자 검색" 
            className="px-4 py-2 border border-gray-200 rounded-lg"
          />
          <select className="px-4 py-2 border border-gray-200 rounded-lg">
            <option>전체</option>
            <option>활성 사용자</option>
            <option>비활성 사용자</option>
          </select>
        </div>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg">
          사용자 추가
        </button>
      </div>

      {/* 사용자 테이블 */}
      <table className="w-full">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">이름</th>
            <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">정수기 ID</th>
            <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">등록일</th>
            <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">상태</th>
            <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">관리</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {/* 샘플 데이터 */}
          {Array(5).fill(null).map((_, i) => (
            <tr key={i}>
              <td className="px-4 py-4 text-sm text-gray-900">사용자 {i + 1}</td>
              <td className="px-4 py-4 text-sm text-gray-600">WP{String(i + 1).padStart(4, '0')}</td>
              <td className="px-4 py-4 text-sm text-gray-600">2024-03-{String(i + 1).padStart(2, '0')}</td>
              <td className="px-4 py-4 text-sm">
                <span className="px-2 py-1 bg-green-50 text-green-600 rounded-full text-xs">
                  활성
                </span>
              </td>
              <td className="px-4 py-4 text-sm">
                <button className="text-blue-600 hover:text-blue-800">수정</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default UserManagement; 
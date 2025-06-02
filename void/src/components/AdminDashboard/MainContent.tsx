import React from 'react';
import Dashboard from './pages/Dashboard';
import UserManagement from './pages/UserManagement';
import PurifierManagement from './pages/PurifierManagement';
import WaterManagement from './pages/WaterManagement';
import VoiceManagement from './pages/VoiceManagement';

interface MainContentProps {
  currentPage: string;
}

const MainContent: React.FC<MainContentProps> = ({ currentPage }) => {
  const renderContent = () => {
    switch (currentPage) {
      case '대시보드':
        return <Dashboard />;
      case '사용자 관리':
        return <UserManagement />;
      case '정수기 관리':
        return <PurifierManagement />;
      case '출수량 관리':
        return <WaterManagement />;
      case 'Voice ID 관리':
        return <VoiceManagement />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="ml-[280px] min-h-screen bg-[#F7F8FA]">
      {/* 상단 헤더 */}
      <div className="h-[72px] px-8 flex items-center justify-between bg-white border-b border-[#DFE3E7]">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-[#1A1F27] tracking-tight">{currentPage}</h2>
          <div className="h-4 w-px bg-[#DFE3E7]" />
          <span className="text-sm text-[#697077]">관리자 대시보드</span>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 text-sm text-[#1428A0] hover:bg-[#1428A0]/[0.04] rounded-lg transition-colors duration-200 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            도움말
          </button>
          <button className="px-3 py-1.5 text-sm text-white font-medium bg-[#1428A0] rounded-lg hover:bg-[#1428A0]/90 transition-colors duration-200 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
            설정
          </button>
        </div>
      </div>
      
      {/* 컨텐츠 영역 */}
      <div className="p-8">
        {renderContent()}
      </div>
    </div>
  );
};

export default MainContent; 
import React, { useState } from 'react';
import { Cog6ToothIcon, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import { useNavigate } from 'react-router-dom';

interface SidebarProps {
  currentPage: string;
  onPageChange: (page: string) => void;
}

// 카테고리 타입을 string | null로 정의
type CategoryType = string | null;

const Sidebar: React.FC<SidebarProps> = ({ currentPage, onPageChange }) => {
  const [expandedCategory, setExpandedCategory] = useState<any>('정수기');
  const navigate = useNavigate();

  const toggleCategory = (category: string) => {
    if (expandedCategory === category) {
      setExpandedCategory(null);
    } else {
      setExpandedCategory(category);
    }
  };

  return (
    <div className="w-[280px] min-h-screen bg-[#F5F7FB] fixed left-0 top-0 border-r border-[#E5E8EC] flex flex-col">
      {/* 헤더 */}
      <div className="h-[80px] px-6 flex items-center gap-3 border-b border-[#E5E8EC] flex-shrink-0">
        <div className="p-2.5 rounded-full bg-[#1428A0]">
          <Cog6ToothIcon className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-[15px] font-bold text-[#1428A0]">
            BESPOKE AI LAB
          </h1>
          <p className="text-[11px] tracking-tight text-[#697077]">Administrator</p>
        </div>
      </div>

      {/* 메뉴 - 스크롤 가능한 영역 */}
      <div className="flex-1 overflow-y-auto pb-4 max-h-[calc(100vh-80px-120px)]">
        <div className="px-4 py-5">
          {/* 비스포크 제품 카테고리 */}
          <div className="mb-4">
            <p className="px-4 text-xs font-medium text-[#697077] uppercase tracking-wider mb-4">비스포크 제품</p>
            
            {/* 정수기 - 토글 가능한 상태 */}
            <div className="mt-3">
              <button 
                className={`w-full h-10 flex items-center justify-between px-4 rounded-[22px] text-[13px] font-medium
                            transition-all duration-200
                            ${expandedCategory === '정수기' 
                              ? 'bg-[#1428A0]/[0.06] text-[#1428A0]' 
                              : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                onClick={() => toggleCategory('정수기')}
              >
                <div className="flex items-center">
                  <WaterIcon color={expandedCategory === '정수기' ? '#1428A0' : '#3C4149'} />
                  <span className="ml-3">정수기</span>
                </div>
                {expandedCategory === '정수기' ? 
                  <ChevronDownIcon className="w-4 h-4" /> : 
                  <ChevronRightIcon className="w-4 h-4" />
                }
              </button>
              {expandedCategory === '정수기' && (
                <div className="ml-4 mt-3 space-y-2">
                  <button 
                    className={`w-full h-9 flex items-center px-4 rounded-[18px] text-[12px] font-medium
                                transition-all duration-200
                                ${currentPage === '정수기 관리' 
                                  ? 'bg-[#1428A0] text-white' 
                                  : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                    onClick={() => {
                      onPageChange('정수기 관리');
                      navigate('/water-purifier');
                    }}
                  >
                    <span>정수기</span>
                  </button>
                  <button 
                    className={`w-full h-9 flex items-center px-4 rounded-[18px] text-[12px]
                                transition-all duration-200
                                ${currentPage === '출수량 관리' 
                                  ? 'bg-[#1428A0] text-white font-medium' 
                                  : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                    onClick={() => onPageChange('출수량 관리')}
                  >
                    <span>출수량</span>
                  </button>
                  <button 
                    className={`w-full h-9 flex items-center px-4 rounded-[18px] text-[12px]
                                transition-all duration-200
                                ${currentPage === 'Voice ID 관리' 
                                  ? 'bg-[#1428A0] text-white font-medium' 
                                  : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                    onClick={() => onPageChange('Voice ID 관리')}
                  >
                    <span>Voice ID</span>
                  </button>

                  {/* MLOps 관리 카테고리 */}
                  <div className="mt-4">
                    <p className="px-4 text-xs font-medium text-[#697077] uppercase tracking-wider mb-3">MLOps 관리</p>

                    {/* 모델 관리 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '모델 관리' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('모델 관리')}
                      >
                        <div className="flex items-center">
                          <ModelIcon color={expandedCategory === '모델 관리' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">모델 관리</span>
                        </div>
                        {expandedCategory === '모델 관리' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '모델 관리' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>모델 버전 관리</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>모델 성능 모니터링</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>모델 배포 상태</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>A/B 테스트 결과</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* 데이터 파이프라인 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '데이터 파이프라인' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('데이터 파이프라인')}
                      >
                        <div className="flex items-center">
                          <PipelineIcon color={expandedCategory === '데이터 파이프라인' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">데이터 파이프라인</span>
                        </div>
                        {expandedCategory === '데이터 파이프라인' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '데이터 파이프라인' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>데이터 수집 현황</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>데이터 품질 모니터링</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>데이터 전처리 파이프라인</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>데이터 버전 관리</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* 인프라 관리 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '인프라 관리' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('인프라 관리')}
                      >
                        <div className="flex items-center">
                          <InfrastructureIcon color={expandedCategory === '인프라 관리' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">인프라 관리</span>
                        </div>
                        {expandedCategory === '인프라 관리' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '인프라 관리' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>컴퓨팅 리소스 사용량</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>스토리지 사용량</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>네트워크 트래픽</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>서비스 상태 모니터링</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* 알림 및 로깅 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '알림 및 로깅' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('알림 및 로깅')}
                      >
                        <div className="flex items-center">
                          <NotificationIcon color={expandedCategory === '알림 및 로깅' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">알림 및 로깅</span>
                        </div>
                        {expandedCategory === '알림 및 로깅' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '알림 및 로깅' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>시스템 알림 설정</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>로그 분석</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>오류 추적</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>성능 메트릭</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* 보안 및 접근 제어 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '보안 및 접근 제어' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('보안 및 접근 제어')}
                      >
                        <div className="flex items-center">
                          <SecurityIcon color={expandedCategory === '보안 및 접근 제어' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">보안 및 접근 제어</span>
                        </div>
                        {expandedCategory === '보안 및 접근 제어' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '보안 및 접근 제어' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>사용자 권한 관리</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>API 키 관리</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>보안 정책 설정</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>감사 로그</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* 실험 관리 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '실험 관리' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('실험 관리')}
                      >
                        <div className="flex items-center">
                          <ExperimentIcon color={expandedCategory === '실험 관리' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">실험 관리</span>
                        </div>
                        {expandedCategory === '실험 관리' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '실험 관리' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>실험 추적</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>하이퍼파라미터 최적화</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>실험 결과 비교</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>재현 가능성 관리</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* API 관리 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === 'API 관리' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('API 관리')}
                      >
                        <div className="flex items-center">
                          <ApiIcon color={expandedCategory === 'API 관리' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">API 관리</span>
                        </div>
                        {expandedCategory === 'API 관리' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === 'API 관리' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>API 엔드포인트 상태</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>API 사용량 통계</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>API 버전 관리</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>API 문서화</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* 보고서 및 분석 */}
                    <div className="mt-2">
                      <button 
                        className={`w-full h-9 flex items-center justify-between px-4 rounded-[18px] text-[12px]
                                    transition-all duration-200
                                    ${expandedCategory === '보고서 및 분석' 
                                      ? 'bg-[#1428A0]/[0.06] text-[#1428A0] font-medium' 
                                      : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
                        onClick={() => toggleCategory('보고서 및 분석')}
                      >
                        <div className="flex items-center">
                          <ReportIcon color={expandedCategory === '보고서 및 분석' ? '#1428A0' : '#3C4149'} />
                          <span className="ml-3">보고서 및 분석</span>
                        </div>
                        {expandedCategory === '보고서 및 분석' ? 
                          <ChevronDownIcon className="w-4 h-4" /> : 
                          <ChevronRightIcon className="w-4 h-4" />
                        }
                      </button>
                      {expandedCategory === '보고서 및 분석' && (
                        <div className="ml-4 mt-2 space-y-1.5">
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>시스템 성능 보고서</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>사용자 행동 분석</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>비용 분석</span>
                          </button>
                          <button className="w-full h-8 flex items-center px-4 rounded-[16px] text-[11px] text-[#3C4149] hover:bg-[#1428A0]/[0.06]">
                            <span>ROI 추적</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* 냉장고 - 비활성화 */}
            <div className="mt-5">
              <div className="w-full h-10 flex items-center justify-between px-4 rounded-[22px] text-[13px]
                            bg-[#F5F7FB] text-[#697077] cursor-not-allowed">
                <div className="flex items-center">
                  <FridgeIcon color="#697077" />
                  <span className="ml-3">냉장고</span>
                </div>
                <ChevronRightIcon className="w-4 h-4" />
              </div>
            </div>
            
            {/* 에어컨 - 비활성화 */}
            <div className="mt-5">
              <div className="w-full h-10 flex items-center justify-between px-4 rounded-[22px] text-[13px]
                            bg-[#F5F7FB] text-[#697077] cursor-not-allowed">
                <div className="flex items-center">
                  <AirconIcon color="#697077" />
                  <span className="ml-3">에어컨</span>
                </div>
                <ChevronRightIcon className="w-4 h-4" />
              </div>
            </div>
            
            {/* 세탁기 - 비활성화 */}
            <div className="mt-5">
              <div className="w-full h-10 flex items-center justify-between px-4 rounded-[22px] text-[13px]
                            bg-[#F5F7FB] text-[#697077] cursor-not-allowed">
                <div className="flex items-center">
                  <WasherIcon color="#697077" />
                  <span className="ml-3">세탁기</span>
                </div>
                <ChevronRightIcon className="w-4 h-4" />
              </div>
            </div>
          </div>

          {/* 기존 메뉴 */}
          <div className="mt-8 mb-4">
            <p className="px-4 text-xs font-medium text-[#697077] uppercase tracking-wider mb-4">시스템</p>
            
            <button 
              className={`w-full h-11 flex items-center px-4 rounded-[22px] text-[13px]
                          transition-all duration-200 relative
                          ${currentPage === '사용자 관리' 
                            ? 'bg-[#1428A0] text-white font-semibold' 
                            : 'text-[#3C4149] hover:bg-[#1428A0]/[0.06]'}`}
              onClick={() => onPageChange('사용자')}
            >
              <UserIcon 
                color={currentPage === '사용자 관리' ? '#fff' : '#3C4149'} 
              />
              <span className="ml-3">사용자</span>
            </button>
          </div>
        </div>
      </div>

      {/* 하단 정보 */}
      <div className="p-5 border-t border-[#E5E8EC] flex-shrink-0">
        <div className="p-4 rounded-2xl bg-white border border-[#E5E8EC]">
          <p className="text-xs text-[#697077] mb-1">현재 버전</p>
          <p className="text-sm font-medium text-[#1428A0]">v3.0.0</p>
        </div>
      </div>
    </div>
  );
};

const DashboardIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 18 18" fill="none">
    <path d="M0 10H8V0H0V10ZM0 18H8V12H0V18ZM10 18H18V8H10V18ZM10 0V6H18V0H10Z" 
          fill={color}/>
  </svg>
);

const UserIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 18 15" fill="none">
    <path d="M9 7.5C10.6569 7.5 12 6.15685 12 4.5C12 2.84315 10.6569 1.5 9 1.5C7.34315 1.5 6 2.84315 6 4.5C6 6.15685 7.34315 7.5 9 7.5Z" 
          fill={color}/>
    <path d="M4 13.5C4 11.0147 6.23858 9 9 9C11.7614 9 14 11.0147 14 13.5" 
          fill={color}/>
  </svg>
);

const WaterIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M10 18C14.4183 18 18 14.4183 18 10C18 5.58172 14.4183 2 10 2C5.58172 2 2 5.58172 2 10C2 14.4183 5.58172 18 10 18Z" 
          stroke={color} strokeWidth="1.5"/>
    <path d="M10 6V14M6 10H14" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const ChartIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M2 2V16H18M4 12L8 8L12 12L16 6" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const VoiceIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M10 2V18M6 6V14M14 8V12M2 10V10M18 8V12" 
          stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const FridgeIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M4 2H16V18H4V2Z" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M4 8H16" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M8 2V18" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const AirconIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M2 6H18V14H2V6Z" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M6 2V18" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M14 2V18" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const WasherIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M2 4H18V16H2V4Z" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M10 8C11.6569 8 13 9.34315 13 11C13 12.6569 11.6569 14 10 14C8.34315 14 7 12.6569 7 11C7 9.34315 8.34315 8 10 8Z" 
          stroke={color} strokeWidth="1.5"/>
  </svg>
);

const ModelIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M10 2L18 6V14L10 18L2 14V6L10 2Z" stroke={color} strokeWidth="1.5"/>
    <path d="M10 6L14 8V12L10 14L6 12V8L10 6Z" stroke={color} strokeWidth="1.5"/>
  </svg>
);

const PipelineIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M2 4H18V16H2V4Z" stroke={color} strokeWidth="1.5"/>
    <path d="M6 8H14" stroke={color} strokeWidth="1.5"/>
    <path d="M6 12H14" stroke={color} strokeWidth="1.5"/>
  </svg>
);

const InfrastructureIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M4 2H16V18H4V2Z" stroke={color} strokeWidth="1.5"/>
    <path d="M4 6H16" stroke={color} strokeWidth="1.5"/>
    <path d="M4 10H16" stroke={color} strokeWidth="1.5"/>
    <path d="M4 14H16" stroke={color} strokeWidth="1.5"/>
  </svg>
);

const NotificationIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M10 2C6.68629 2 4 4.68629 4 8V14L2 16H18L16 14V8C16 4.68629 13.3137 2 10 2Z" stroke={color} strokeWidth="1.5"/>
    <path d="M10 18V18.01" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const SecurityIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M10 2L18 6V10C18 14.4183 14.4183 18 10 18C5.58172 18 2 14.4183 2 10V6L10 2Z" stroke={color} strokeWidth="1.5"/>
    <path d="M7 10L9 12L13 8" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ExperimentIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M4 4H16V16H4V4Z" stroke={color} strokeWidth="1.5"/>
    <path d="M8 8L12 12" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M12 8L8 12" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const ApiIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M10 2L18 6V14L10 18L2 14V6L10 2Z" stroke={color} strokeWidth="1.5"/>
    <path d="M10 6L14 8V12L10 14L6 12V8L10 6Z" stroke={color} strokeWidth="1.5"/>
  </svg>
);

const ReportIcon = ({ color }: { color: string }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
    <path d="M4 2H16V18H4V2Z" stroke={color} strokeWidth="1.5"/>
    <path d="M6 6H14" stroke={color} strokeWidth="1.5"/>
    <path d="M6 10H14" stroke={color} strokeWidth="1.5"/>
    <path d="M6 14H10" stroke={color} strokeWidth="1.5"/>
  </svg>
);

export default Sidebar; 
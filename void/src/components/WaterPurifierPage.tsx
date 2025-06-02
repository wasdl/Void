import React, { useState, useRef, useEffect } from 'react';

const WaterPurifierPage: React.FC = () => {
  const [isZoomed, setIsZoomed] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const [showBixbyVideo, setShowBixbyVideo] = useState(false);
  const [clickCount, setClickCount] = useState(0);
  const [selectedTemp, setSelectedTemp] = useState('미온수');
  const [selectedDegree, setSelectedDegree] = useState('40');
  const [selectedVolume, setSelectedVolume] = useState('120ml');
  const [hotWaterMode, setHotWaterMode] = useState(0);
  const [isVoiceRegistration, setIsVoiceRegistration] = useState(false);
  const [voiceCount, setVoiceCount] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceComplete, setIsVoiceComplete] = useState(false);
  const [isPersonalization, setIsPersonalization] = useState(false);
  const [isNameConfirmation, setIsNameConfirmation] = useState(false);
  const [tempName, setTempName] = useState('');
  const [isNameConfirmed, setIsNameConfirmed] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const offAudioRef = useRef<HTMLAudioElement | null>(null);
  const [isBixbyComplete, setIsBixbyComplete] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [videoUrl, setVideoUrl] = useState('/video/bixby.mp4');
  const [isDispensing, setIsDispensing] = useState(false);
  const [bixbyResponse, setBixbyResponse] = useState('');

  const handleTempClick = (temp: string) => {
    if (isVoiceRegistration) return;

    if (temp === '온수') {
      setHotWaterMode((prev) => {
        switch (prev) {
          case 0:
            setSelectedTemp('녹차');
            setSelectedDegree('75');
            return 1;
          case 1:
            setSelectedTemp('커피');
            setSelectedDegree('85');
            return 2;
          case 2:
            setSelectedVolume('1잔');
            setSelectedTemp('브루잉');
            setSelectedDegree('보통');
            return 3;
          case 3:
            setSelectedTemp('미온수');
            setSelectedDegree('40');
            return 0;
          default:
            setSelectedTemp('미온수');
            setSelectedDegree('40');
            return 0;
        }
      });
    } else if (temp === '냉수') {
      setSelectedTemp('냉수');
      setHotWaterMode(0);
    } else if (temp === '정수' && !isVoiceRegistration) {
      setSelectedTemp('정수');
      setHotWaterMode(0);
    }
  };

  const handlePurifiedWaterClick = () => {
    if (isVoiceRegistration) {
      if (!isVoiceComplete) {
        // 목소리 녹음 진행
        if (voiceCount < 10) {
          setIsRecording(true);
          setTimeout(() => {
            const newCount = voiceCount + 1;
            setVoiceCount(newCount);
            setIsRecording(false);
            if (newCount === 10) {
              setIsVoiceComplete(true);
            }
          }, 2000);
        }
      } else if (isVoiceComplete && !isNameConfirmation) {
        setTempName('지해');
        setTimeout(() => {
          setIsNameConfirmation(true);
        }, 3000);
      } else if (isNameConfirmation) {
        setIsVoiceComplete(false);
        setIsNameConfirmation(false);
        setTempName('');
        setVoiceCount(0);
      }
    } else {
      // 일반 정수 버튼 동작
      setSelectedTemp('정수');
      setHotWaterMode(0);
    }
  };

  const handleVolumeClick = () => {
    if (isVoiceRegistration) {
      if (isNameConfirmation && !isNameConfirmed) {
        setIsNameConfirmed(true);
        setTimeout(() => {
          handleVoiceRegistrationClose();
        }, 2000);
        return;
      }
    }

    // 일반 출수량 버튼 동작
    if (!isVoiceRegistration) {
      switch (selectedVolume) {
        case '120ml':
          setSelectedVolume('260ml');
          break;
        case '260ml':
          setSelectedVolume('500ml');
          break;
        case '500ml':
          setSelectedVolume('1000ml');
          break;
        case '1000ml':
          setSelectedVolume('연속출수');
          break;
        case '연속출수':
          setSelectedVolume('120ml');
          break;
        default:
          setSelectedVolume('120ml');
      }
    }
  };

  const handleClick = () => {
    if (clickCount === 0) {
      setIsZoomed(true);
      setTimeout(() => {
        setClickCount(1);
      }, 1000);
    } else if (clickCount === 1) {
      setClickCount(2);
      setShowControls(true);
    }
  };

  const handleReset = () => {
    setShowControls(false);
    setClickCount(0);
    setIsZoomed(false);
    setSelectedTemp('미온수');
    setSelectedDegree('40');
    setSelectedVolume('120ml');
    setHotWaterMode(0);
    setShowBixbyVideo(false);
    setIsBixbyComplete(false);
    setVideoUrl('/video/bixby.mp4');
    setIsVoiceRegistration(false);
    setVoiceCount(0);
    setIsVoiceComplete(false);
    setIsNameConfirmation(false);
    setTempName('');
    setIsNameConfirmed(false);
    setIsDispensing(false);
    setBixbyResponse('');
  };

  const handlePurifiedWaterMouseDown = () => {
    if (isVoiceRegistration) {
      return;
    }

    setTimeout(() => {
      setIsVoiceRegistration(true);
    }, 3000);
  };

  const handlePurifiedWaterMouseUp = () => {
    // 마우스 업 이벤트 처리
  };

  const handleVoiceRegistrationClose = () => {
    setIsVoiceRegistration(false);
    setVoiceCount(0);
    setIsVoiceComplete(false);
    setIsNameConfirmation(false);
    setTempName('');
    setIsNameConfirmed(false);
  };

  const handleVoiceButtonPress = () => {
    if (voiceCount < 10 && !isVoiceComplete) {
      setIsRecording(true);
      setTimeout(() => {
        const newCount = voiceCount + 1;
        setVoiceCount(newCount);
        setIsRecording(false);
        if (newCount === 10) {
          setIsVoiceComplete(true);
          setIsNameConfirmation(false);
          setTempName('');
          setIsNameConfirmed(false);
        }
      }, 2000);
    }
  };

  const handleVoiceButtonRelease = () => {
    setIsRecording(false);
  };

  const handleVolumeButtonMouseDown = () => {
    if (isVoiceRegistration && isNameConfirmation) {
      setTimeout(() => {
        setIsNameConfirmation(false);
        setTempName('');
      }, 3000);
    }
  };

  const handleVolumeButtonMouseUp = () => {
    // 볼륨 버튼 마우스 업 이벤트 처리
  };

  const handleVoiceRegistration = () => {
    if (!showBixbyVideo) {
      setShowBixbyVideo(true);
      if (audioRef.current) {
        audioRef.current.play();
      }
    }
  };

  const handleBixbyComplete = () => {
    if (offAudioRef.current) {
      offAudioRef.current.play();
    }
    setVideoUrl('/video/bixby_complete.mp4');
    setShowBixbyVideo(true);
    setIsBixbyComplete(true);
  };

  const handlePersonalization = (name: string, volume: string) => {
    setShowBixbyVideo(false);
    setVideoUrl('/video/bixby.mp4');
    setSelectedTemp(name);
    setSelectedVolume(volume);
    setBixbyResponse(`안녕하세요!\n${name}님\n${volume} 출수를\n시작합니다.`);
    setIsPersonalization(true);
    setIsDispensing(true);
    
    setTimeout(() => {
      setSelectedTemp('미온수');
      setSelectedDegree('40');
      setIsPersonalization(false);
      setIsDispensing(false);
      setBixbyResponse('');
    }, 3000);
  };

  useEffect(() => {
    audioRef.current = new Audio('/sounds/on.m4a');
    offAudioRef.current = new Audio('/sounds/off.m4a');
  }, []);

  useEffect(() => {
    handleReset();
  }, []);

  useEffect(() => {
    console.log('목소리 등록 상태:', isVoiceRegistration);
  }, [isVoiceRegistration]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!showControls) return;

      if (e.key.toLowerCase() === 'a') {
        handleVoiceRegistration();
      } else if (e.key.toLowerCase() === 's') {
        handleBixbyComplete();
      } else if (e.key.toLowerCase() === 'q') {
        handlePersonalization('배지해', '320ml');
      } else if (e.key.toLowerCase() === 'w') {
        handlePersonalization('임창현', '230ml');
      } else if (e.key.toLowerCase() === 'e') {
        handlePersonalization('제갈민', '170ml');
      } else if (e.key.toLowerCase() === 'r') {
        handlePersonalization('박성재', '120ml');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [showControls, isBixbyComplete]);

  return (
    <div className="w-full h-screen overflow-hidden">
      <div className={`relative transform-gpu translate-x-0`}>
        {clickCount === 0 ? (
          <div
            className={`w-full h-screen transform-gpu ${isZoomed ? 'transition-all duration-1000 ease-in-out scale-[1.3] origin-[60%_100%]' : 'scale-100'
              }`}
            onClick={handleClick}
          >
            <img
              src={'/images/water_purifier_O.png'}
              alt="BESPOKE 정수기"
              className="w-full h-full object-cover"
            />
            {!isZoomed ? (
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-8">
                <h1 className="text-3xl font-bold text-white">BESPOKE 정수기</h1>
                <p className="text-white/80 mt-2 text-lg">음성 인식 기능을 통해 쉽고 빠르게 원하는 물을 뽑아보세요</p>
              </div>
            ) : null}
          </div>
        ) : (
          <div
            className="w-full h-screen scale-[1.3] origin-[60%_100%]"
            onClick={handleClick}
          >
            <img
              src={'/images/water_purifier_O.png'}
              alt="BESPOKE 정수기"
              className={`w-full h-full object-cover ${clickCount >= 2 && isZoomed
                  ? 'duration-1000 ease-in-out w-2/3 -translate-x-[12%]'
                  : 'w-full translate-x-0'
                }`}
            />
          </div>
        )}
      </div>

      {showControls && (
        <div
          className="w-1/4 h-full bg-black absolute top-0 right-0 flex items-center justify-center transition-all duration-1000 ease-in-out transform translate-x-full"
          style={{ transform: showControls ? 'translateX(0)' : undefined }}
        >
          <div className="w-full h-full relative flex flex-col rounded-b-[2rem]">
            <button
              onClick={isVoiceRegistration ? handleVoiceRegistrationClose : handleReset}
              className="py-4 px-8 flex items-center space-x-2 text-white/70 hover:text-white transition-colors group"
            >
              <svg
                className="w-6 h-6 transform transition-transform group-hover:-translate-x-1"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              <span>{isVoiceRegistration ? '이전으로' : '처음으로 돌아가기'}</span>
            </button>

            <div className="flex-1 m-8 bg-gradient-to-br from-white/15 to-black/70 flex flex-col rounded-[2rem] border border-white/20">
              {!isVoiceRegistration ? (
                <>
                  <div className="flex-1 flex items-center justify-center">
                    <div className="relative bg-black/80 px-6 py-4 rounded-lg text-center h-[150px] w-[130px] flex flex-col justify-center overflow-hidden">
                      {!showBixbyVideo && (
                        <>
                          <div className="h-[40px] flex items-center justify-center">
                            {['미온수', '녹차', '커피', '보통 브루잉 1잔'].includes(selectedTemp) ? (
                              <div className="text-2xl font-medium text-white flex items-center space-x-2">
                                <span>{selectedDegree}°</span>
                              </div>
                            ) : selectedTemp === '배지해' || selectedTemp === '임창현' || selectedTemp === '제갈민' || selectedTemp === '박성재' ? (
                              <div className="text-lg text-blue-400">
                                {selectedDegree}
                              </div>
                            ) : (
                              <div className="flex items-center justify-center">
                                <svg className="w-8 h-8 text-white/60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                  <path d="M12 6.5C12 6.5 16 10.5 16 14.5C16 17.5376 14.2091 20 12 20C9.79086 20 8 17.5376 8 14.5C8 10.5 12 6.5 12 6.5Z" stroke="currentColor" strokeWidth="1.5" />
                                </svg>
                              </div>
                            )}
                          </div>
                          <div className="text-lg text-white h-[30px] flex items-center justify-center">{selectedTemp}</div>
                          <div className="text-lg text-white h-[30px] flex items-center justify-center whitespace-nowrap">{selectedVolume}</div>
                          {isDispensing && (
                            <div className="absolute inset-0 bg-black/80 rounded-lg flex items-center justify-center">
                              <div className="text-center p-4">
                                <div className="text-white text-base font-medium whitespace-pre-line">{bixbyResponse}</div>
                              </div>
                            </div>
                          )}
                        </>
                      )}
                      {showBixbyVideo && (
                        <div className="absolute inset-0 z-50">
                          <video
                            ref={videoRef}
                            className="w-full h-full object-cover rounded-lg"
                            autoPlay
                            loop
                            muted
                            playsInline
                            src={videoUrl}
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="space-y-12 mb-12">
                    <div className="flex px-8">
                      <button
                        onClick={() => handleTempClick('온수')}
                        className={`flex-1 transition-colors text-base ${['미온수', '녹차', '커피', '보통 브루잉 1잔'].includes(selectedTemp) ? 'text-white' : 'text-white/60 hover:text-white'
                          }`}
                      >
                        온수
                      </button>
                      <button
                        onClick={() => handleTempClick('냉수')}
                        className={`flex-1 transition-colors text-base ${selectedTemp === '냉수' ? 'text-white' : 'text-white/60 hover:text-white'
                          }`}
                      >
                        냉수
                      </button>
                    </div>

                    <div className="flex px-8">
                      <button
                        onMouseDown={handlePurifiedWaterMouseDown}
                        onMouseUp={handlePurifiedWaterMouseUp}
                        onMouseLeave={handlePurifiedWaterMouseUp}
                        onClick={() => handleTempClick('정수')}
                        className={`flex-1 transition-colors text-base ${selectedTemp === '정수' ? 'text-white' : 'text-white/60 hover:text-white'
                          }`}
                      >
                        정수
                      </button>
                      <button
                        onClick={handleVolumeClick}
                        className="flex-1 text-white/60 hover:text-white transition-colors text-base"
                      >
                        출수량
                      </button>
                    </div>

                    <div className="flex justify-center mt-8">
                      <button className={`w-24 h-24 rounded-full border-2 ${isPersonalization
                          ? 'border-blue-400 bg-blue-400/20'
                          : 'border-white/80 hover:bg-white/5'
                        } flex items-center justify-center transition-colors group`}>
                        <svg className={`w-24 h-24 ${isPersonalization ? 'text-blue-400' : 'text-white/60'
                          }`} viewBox="0 1 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 8C12 8 15 11 15 14C15 16.2091 13.6569 18 12 18C10.3431 18 9 16.2091 9 14C9 11 12 8 12 8Z" stroke="currentColor" strokeWidth="0.5" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex-1 flex items-center justify-center">
                    <div className="bg-black/80 py-2 rounded-lg text-center h-[150px] w-[130px] flex flex-col justify-center">
                      <div className="flex flex-col items-center">
                        {!isVoiceComplete ? (
                          <>
                            <svg className={`w-8 h-8 ${isRecording ? 'text-blue-400' : 'text-white/60'} mb-2 transition-colors duration-200`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                            </svg>
                            <div className="text-sm text-white mb-1">두비두</div>
                            <div className="text-xs text-white/60">출수 버튼을 누르고</div>
                            <div className="text-xs text-white/60">말씀해주세요</div>
                            <div className={`text-xs ${isRecording ? 'text-blue-400' : 'text-white/40'} mt-2 transition-colors duration-200`}>
                              {voiceCount}/10
                            </div>
                          </>
                        ) : !isNameConfirmed ? (
                          <>
                            <svg className="w-8 h-8 text-blue-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            {!isNameConfirmation ? (
                              <>
                                <div className="text-sm text-white mb-1">이름설정</div>
                                <div className="text-xs text-white/60">출수 버튼을 누르고</div>
                                <div className="text-xs text-white/60">이름을 말씀해주세요</div>
                                <div className="text-xs text-blue-400 mt-2">
                                  음성 인식 준비 완료
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="text-sm text-white mb-1">이름확인</div>
                                <div className="text-xs text-white/60">{tempName}님이 맞으실까요?</div>
                                <div className="text-xs text-white/60 mt-2">
                                  등록을 위해<br />출수량 버튼을 눌러주세요
                                </div>
                                <div className="text-xs text-white/60 mt-1">
                                  다시 녹음을 위해<br />출수 버튼을 눌러주세요
                                </div>
                              </>
                            )}
                          </>
                        ) : (
                          <>
                            <svg className="w-8 h-8 text-green-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <div className="text-sm text-white mb-1">등록완료</div>
                            <div className="text-xs text-white/60">{tempName}님의</div>
                            <div className="text-xs text-white/60">목소리가 등록되었습니다</div>
                            <div className="text-xs text-blue-400 mt-2">
                              이제 음성으로 제어하세요
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-12 mb-12">
                    <div className="flex px-8">
                      <button className="flex-1 text-white/20 text-base">온수</button>
                      <button className="flex-1 text-white/20 text-base">냉수</button>
                    </div>

                    <div className="flex px-8">
                      <button className="flex-1 text-white/20 text-base">
                        <div className="flex flex-col items-center">
                          <div>정수</div>
                          <div className="mt-1 text-xs text-white/20">
                            목소리 등록<br />
                            (3초)
                          </div>
                        </div>
                      </button>
                      <button
                        onClick={handleVolumeClick}
                        onMouseDown={handleVolumeButtonMouseDown}
                        onMouseUp={handleVolumeButtonMouseUp}
                        onMouseLeave={handleVolumeButtonMouseUp}
                        onTouchStart={handleVolumeButtonMouseDown}
                        onTouchEnd={handleVolumeButtonMouseUp}
                        className={`flex-1 transition-colors text-base ${isVoiceComplete ? 'text-white/20' : 'text-white/20'
                          }`}
                      >
                        <div className="flex flex-col items-center">
                          <div>출수량</div>
                          <div className="mt-1 text-xs text-white/40">
                            {isVoiceComplete && !isNameConfirmed ? (
                              isNameConfirmation ? "확인" : "이름설정 대기"
                            ) : (
                              <>
                                필터교체<br />
                                (3초)
                              </>
                            )}
                          </div>
                        </div>
                      </button>
                    </div>

                    <div className="flex justify-center mt-8">
                      <button
                        onMouseDown={!isVoiceComplete ? handleVoiceButtonPress : undefined}
                        onMouseUp={!isVoiceComplete ? handleVoiceButtonRelease : undefined}
                        onMouseLeave={!isVoiceComplete ? handleVoiceButtonRelease : undefined}
                        onTouchStart={!isVoiceComplete ? handleVoiceButtonPress : undefined}
                        onTouchEnd={!isVoiceComplete ? handleVoiceButtonRelease : undefined}
                        disabled={isVoiceComplete}
                        className={`w-24 h-24 rounded-full border-2 ${isVoiceComplete || isRecording || isPersonalization
                            ? 'bg-white/20 border-white/20'
                            : 'border-white/80 hover:bg-white/5'
                          } flex items-center justify-center transition-all duration-200 group`}
                      >
                        <svg className={`w-24 h-24 ${isRecording ? 'text-blue-400' : 'text-white/60'} transition-colors duration-200`} viewBox="0 1 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 8C12 8 15 11 15 14C15 16.2091 13.6569 18 12 18C10.3431 18 9 16.2091 9 14C9 11 12 8 12 8Z" stroke="currentColor" strokeWidth="0.5" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WaterPurifierPage; 
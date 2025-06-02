// 가정 그룹 타입 정의
export interface FamilyGroup {
  familyId: string;
  members: number;
}

// 클러스터링 결과 타입 정의
interface ClusteringResult {
  voiceId: string;
  similarity: number;
}

// 로그 데이터 타입 정의
export interface LogData {
  timestamp: string;
  action: string;
  result: string;
  clusteringResults?: ClusteringResult[];
}

// 음성 샘플 타입 정의
export interface VoiceSample {
  id: string;
  familyId: string;
  gender: '남성' | '여성';
  age: number;
  registeredAt: string;
  accuracy: number;
  logs: LogData[];
}

// 가정 그룹 더미 데이터 확장
export const familyGroups: FamilyGroup[] = [
  { familyId: 'HOME_001', members: 4 },
  { familyId: 'HOME_002', members: 3 },
  { familyId: 'HOME_003', members: 4 },
  { familyId: 'HOME_004', members: 3 },
  { familyId: 'HOME_005', members: 4 },
  { familyId: 'HOME_006', members: 5 },
  { familyId: 'HOME_007', members: 3 },
  { familyId: 'HOME_008', members: 4 },
  { familyId: 'HOME_009', members: 2 },
  { familyId: 'HOME_010', members: 4 },
];

// VOICE ID 배열 확장
const voiceIds = {
  'HOME_001': ['VOICE_001', 'VOICE_002', 'VOICE_003', 'VOICE_004'],
  'HOME_002': ['VOICE_005', 'VOICE_006', 'VOICE_007'],
  'HOME_003': ['VOICE_008', 'VOICE_009', 'VOICE_010', 'VOICE_011'],
  'HOME_004': ['VOICE_012', 'VOICE_013', 'VOICE_014'],
  'HOME_005': ['VOICE_015', 'VOICE_016', 'VOICE_017', 'VOICE_018'],
  'HOME_006': ['VOICE_019', 'VOICE_020', 'VOICE_021', 'VOICE_022', 'VOICE_023'],
  'HOME_007': ['VOICE_024', 'VOICE_025', 'VOICE_026'],
  'HOME_008': ['VOICE_027', 'VOICE_028', 'VOICE_029', 'VOICE_030'],
  'HOME_009': ['VOICE_031', 'VOICE_032'],
  'HOME_010': ['VOICE_033', 'VOICE_034', 'VOICE_035', 'VOICE_036']
};

// 랜덤 로그 생성 함수
const generateRandomLogs = (count: number, familyId: string): LogData[] => {
  const logs: LogData[] = [];
  const currentDate = new Date();
  
  // 해당 가족의 VOICE ID 목록 가져오기
  const familyVoices = voiceIds[familyId as keyof typeof voiceIds];
  
  for (let i = 0; i < count; i++) {
    const randomDays = Math.floor(Math.random() * 7);
    const randomHours = Math.floor(Math.random() * 24);
    const randomMinutes = Math.floor(Math.random() * 60);
    const logDate = new Date(currentDate);
    logDate.setDate(logDate.getDate() - randomDays);
    logDate.setHours(randomHours, randomMinutes);

    // 메인 유사도 (성공 기준: 85% 이상)
    const mainSimilarity = Math.random() * 20 + 80; // 70-100% 범위
    const success = mainSimilarity >= 85;
    let sums = 100;
    // 클러스터링 결과 생성
    const clusteringResults = familyVoices.map(voiceId => {
      let similarity;
      if (success && voiceId === familyVoices[0]) {
        similarity = mainSimilarity;
      }else if (success && voiceId === familyVoices[-1]) {
        similarity=sums;
      }else {
        similarity = Math.random() * sums; // 30-70% 범위
      }
      sums-=similarity;
      return {
        voiceId,
        similarity: Number(similarity.toFixed(2))
      };
    }).sort((a, b) => b.similarity - a.similarity);

    logs.push({
      timestamp: logDate.toLocaleString(),
      action: '인증 시도',
      result: success ? '성공' : '실패',
      clusteringResults
    });
  }

  return logs.sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
};

// 음성 샘플 더미 데이터 확장
export const voiceSamples: VoiceSample[] = [
  // HOME_001 가족
  {
    id: 'VOICE_001',
    familyId: 'HOME_001',
    gender: '남성',
    age: 45,
    registeredAt: '2024-02-15',
    accuracy: 78.9,
    logs: generateRandomLogs(5, 'HOME_001')
  },
  {
    id: 'VOICE_002',
    familyId: 'HOME_001',
    gender: '여성',
    age: 42,
    registeredAt: '2024-02-15',
    accuracy: 80.1,
    logs: generateRandomLogs(4, 'HOME_001')
  },
  {
    id: 'VOICE_003',
    familyId: 'HOME_001',
    gender: '남성',
    age: 15,
    registeredAt: '2024-02-16',
    accuracy: 75.4,
    logs: generateRandomLogs(6, 'HOME_001')
  },
  {
    id: 'VOICE_004',
    familyId: 'HOME_001',
    gender: '여성',
    age: 12,
    registeredAt: '2024-02-16',
    accuracy: 73.1,
    logs: generateRandomLogs(3, 'HOME_001')
  },

  // HOME_002 가족
  {
    id: 'VOICE_005',
    familyId: 'HOME_002',
    gender: '남성',
    age: 38,
    registeredAt: '2024-02-20',
    accuracy: 72.1,
    logs: generateRandomLogs(4, 'HOME_002')
  },
  {
    id: 'VOICE_006',
    familyId: 'HOME_002',
    gender: '여성',
    age: 35,
    registeredAt: '2024-02-20',
    accuracy: 71.9,
    logs: generateRandomLogs(5, 'HOME_002')
  },
  {
    id: 'VOICE_007',
    familyId: 'HOME_002',
    gender: '여성',
    age: 8,
    registeredAt: '2024-02-21',
    accuracy: 81.1,
    logs: generateRandomLogs(3, 'HOME_002')
  },

  // HOME_003 가족
  {
    id: 'VOICE_008',
    familyId: 'HOME_003',
    gender: '남성',
    age: 52,
    registeredAt: '2024-02-22',
    accuracy: 77.4,
    logs: generateRandomLogs(6, 'HOME_003')
  },
  {
    id: 'VOICE_009',
    familyId: 'HOME_003',
    gender: '여성',
    age: 48,
    registeredAt: '2024-02-22',
    accuracy:69.1,
    logs: generateRandomLogs(4, 'HOME_003')
  },
  {
    id: 'VOICE_010',
    familyId: 'HOME_003',
    gender: '남성',
    age: 22,
    registeredAt: '2024-02-23',
    accuracy: 81.8,
    logs: generateRandomLogs(5, 'HOME_003')
  },
  {
    id: 'VOICE_011',
    familyId: 'HOME_003',
    gender: '여성',
    age: 19,
    registeredAt: '2024-02-23',
    accuracy: 71.9,
    logs: generateRandomLogs(4, 'HOME_003')
  },

  // HOME_004 가족
  {
    id: 'VOICE_012',
    familyId: 'HOME_004',
    gender: '여성',
    age: 41,
    registeredAt: '2024-02-25',
    accuracy: 75.1,
    logs: generateRandomLogs(5, 'HOME_004')
  },
  {
    id: 'VOICE_013',
    familyId: 'HOME_004',
    gender: '남성',
    age: 16,
    registeredAt: '2024-02-25',
    accuracy: 81.1,
    logs: generateRandomLogs(4, 'HOME_004')
  },
  {
    id: 'VOICE_014',
    familyId: 'HOME_004',
    gender: '여성',
    age: 14,
    registeredAt: '2024-02-25',
    accuracy: 84.4,
    logs: generateRandomLogs(3, 'HOME_004')
  },

  // HOME_005 가족
  {
    id: 'VOICE_015',
    familyId: 'HOME_005',
    gender: '남성',
    age: 47,
    registeredAt: '2024-02-26',
    accuracy: 70.1,
    logs: generateRandomLogs(6, 'HOME_005')
  },
  {
    id: 'VOICE_016',
    familyId: 'HOME_005',
    gender: '여성',
    age: 45,
    registeredAt: '2024-02-26',
    accuracy: 72.1,
    logs: generateRandomLogs(5, 'HOME_005')
  },
  {
    id: 'VOICE_017',
    familyId: 'HOME_005',
    gender: '남성',
    age: 18,
    registeredAt: '2024-02-27',
    accuracy: 69.9,
    logs: generateRandomLogs(4, 'HOME_005')
  },
  {
    id: 'VOICE_018',
    familyId: 'HOME_005',
    gender: '여성',
    age: 16,
    registeredAt: '2024-02-27',
    accuracy: 75.9,
    logs: generateRandomLogs(3, 'HOME_005')
  },

  // HOME_006 가족 (대가족)
  {
    id: 'VOICE_019',
    familyId: 'HOME_006',
    gender: '남성',
    age: 55,
    registeredAt: '2024-02-28',
    accuracy: 74.4,
    logs: generateRandomLogs(5, 'HOME_006')
  },
  {
    id: 'VOICE_020',
    familyId: 'HOME_006',
    gender: '여성',
    age: 52,
    registeredAt: '2024-02-28',
    accuracy: 88.1,
    logs: generateRandomLogs(4, 'HOME_006')
  },
  {
    id: 'VOICE_021',
    familyId: 'HOME_006',
    gender: '남성',
    age: 25,
    registeredAt: '2024-02-28',
    accuracy: 75.5,
    logs: generateRandomLogs(6, 'HOME_006')
  },
  {
    id: 'VOICE_022',
    familyId: 'HOME_006',
    gender: '여성',
    age: 23,
    registeredAt: '2024-02-28',
    accuracy: 73.4,
    logs: generateRandomLogs(5, 'HOME_006')
  },
  {
    id: 'VOICE_023',
    familyId: 'HOME_006',
    gender: '여성',
    age: 20,
    registeredAt: '2024-02-28',
    accuracy: 81.8,
    logs: generateRandomLogs(4, 'HOME_006')
  },

  // ... HOME_007 ~ HOME_010 데이터도 비슷한 패턴으로 추가 ...
];

// 성별 통계 데이터 업데이트
export const genderData = [
  { name: '남성', value: 1634, color: '#3B82F6' },
  { name: '여성', value: 1589, color: '#EC4899' }
]; 
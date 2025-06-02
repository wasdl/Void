import React, { useState } from 'react';
import Sidebar from './Sidebar';
import MainContent from './MainContent';

const AdminDashboard: React.FC = () => {
  const [currentPage, setCurrentPage] = useState('대시보드');

  return (
    <div className="min-h-screen bg-white">
      <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
      <MainContent currentPage={currentPage} />
    </div>
  );
};

export default AdminDashboard; 
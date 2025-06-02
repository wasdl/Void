import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AdminDashboard from './components/AdminDashboard/index';
import WaterPurifierPage from './components/WaterPurifierPage';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/admin-dashboard" element={<AdminDashboard />} />
        <Route path="/" element={<WaterPurifierPage />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
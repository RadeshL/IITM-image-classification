import React from 'react';
import styled from 'styled-components';

const ResultsCards = ({ results }) => {
  // Default mock results (replace later with real prediction)
  const defaultResults = {
    bolt: 0,
    locatingPin: 0,
    nut: 0,
    washer: 0,
  };

  const data = results || defaultResults;

  return (
    <StyledWrapper>
      <div className="cards">
        <div className="card bolt">
          <p className="tip">Bolt</p>
          <p className="second-text">Count: {data.bolt}</p>
        </div>
        <div className="card locating-pin">
          <p className="tip">Locating Pin</p>
          <p className="second-text">Count: {data.locatingPin}</p>
        </div>
        <div className="card nut">
          <p className="tip">Nut</p>
          <p className="second-text">Count: {data.nut}</p>
        </div>
        <div className="card washer">
          <p className="tip">Washer</p>
          <p className="second-text">Count: {data.washer}</p>
        </div>
      </div>
    </StyledWrapper>
  );
};

const StyledWrapper = styled.div`
  .cards {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .card {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    text-align: center;
    height: 30px;
    width: 300px;
    border-radius: 16px;
    color: white;
    cursor: default;
    transition: all 0.4s ease;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }

  .bolt { background: linear-gradient(135deg, rgba(82, 39, 255, 0.7), rgba(120, 80, 255, 0.5)); }
  .locating-pin { background: linear-gradient(135deg, rgba(140, 60, 255, 0.7), rgba(180, 100, 255, 0.5)); }
  .nut { background: linear-gradient(135deg, rgba(200, 80, 255, 0.7), rgba(240, 120, 255, 0.5)); }
  .washer { background: linear-gradient(135deg, rgba(255, 100, 252, 0.7), rgba(255, 140, 252, 0.5)); }

  .card p.tip {
    font-size: 1rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 0 15px rgba(255, 255, 255, 0.6);
  }

  .card p.second-text {
    font-size: 1rem;
    margin: 8px 0 0;
    opacity: 0.9;
  }

  .card:hover {
    transform: scale(1.08) translateY(-5px);
    box-shadow: 0 15px 40px rgba(82, 39, 255, 0.5);
  }

  .cards:hover > .card:not(:hover) {
    filter: blur(8px);
    transform: scale(0.92);
    opacity: 0.7;
  }
`;

export default ResultsCards;
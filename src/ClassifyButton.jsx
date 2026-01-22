import React from 'react';
import styled from 'styled-components';

const ClassifyButton = ({ onClick, disabled }) => {
  return (
    <StyledWrapper>
      <button onClick={onClick} disabled={disabled}>
        CLASSIFY
      </button>
    </StyledWrapper>
  );
};

const StyledWrapper = styled.div`
  button {
    --purple: #5227FF;
    --pink: #FF9FFC;
    font-size: 1.5rem;
    padding: 0.5em 1em;
    letter-spacing: 0.1em;
    position: relative;
    font-family: 'Orbitron', 'Courier New', monospace; /* Futuristic font */
    font-weight: 800;
    border-radius: 0.8em;
    overflow: hidden;
    transition: all 0.4s;
    line-height: 1.4em;
    border: 3px solid var(--pink);
    background: linear-gradient(to right, 
      rgba(82, 39, 255, 0.15) 1%, 
      transparent 40%, 
      transparent 60%, 
      rgba(255, 159, 252, 0.15) 100%
    );
    color: #FF9FFC;
    text-shadow: 0 0 15px rgba(255, 159, 252, 0.8);
    box-shadow: 
      inset 0 0 20px rgba(82, 39, 255, 0.3), 
      0 0 20px 5px rgba(255, 159, 252, 0.2);

    cursor: ${props => (props.disabled ? 'not-allowed' : 'pointer')};
    opacity: ${props => (props.disabled ? 0.5 : 1)};
  }

  button:hover:not(:disabled) {
    color: #ffffff;
    text-shadow: 0 0 25px rgba(255, 255, 255, 0.9);
    box-shadow: 
      inset 0 0 25px rgba(82, 39, 255, 0.6), 
      0 0 30px 8px rgba(255, 159, 252, 0.4);
    transform: translateY(-3px);
  }

  button:before {
    content: "";
    position: absolute;
    left: 6em;
    width: 2em;
    height: 30px;
    top: 0;
    transition: transform 0.6s ease-in-out;
    background: linear-gradient(to right, 
      transparent 1%, 
      rgba(255, 159, 252, 0.3) 40%, 
      rgba(82, 39, 255, 0.3) 60%, 
      transparent 100%
    );
  }

  button:hover:before:not(:disabled) {
    transform: translateX(20em);
  }
`;

export default ClassifyButton;
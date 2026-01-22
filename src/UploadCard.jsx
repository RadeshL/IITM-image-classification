import React, { useState } from 'react';
import styled from 'styled-components';

const UploadCard = ({ onImageUpload }) => {
  const [imagePreview, setImagePreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);  // Store actual File

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        setImageFile(file);

        const reader = new FileReader();
        reader.onload = (event) => {
          const dataUrl = event.target.result;
          setImagePreview(dataUrl);
          if (onImageUpload) onImageUpload(file, dataUrl);  // Pass both
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  };

  return (
    <StyledWrapper onClick={handleClick}>
      <div className="card">
        {imagePreview ? (
          <img src={imagePreview} alt="Preview" className="preview-image" />
        ) : (
          <div className="placeholder">
            <span className="plus-icon">+</span>
            <p>Click to Upload Image</p>
          </div>
        )}
      </div>
    </StyledWrapper>
  );
};

// Same styled components as before...
const StyledWrapper = styled.div`
  .card {
    position: relative;
    width: 300px;
    height: 300px;
    background: rgba(30, 15, 60, 0.6);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    font-weight: bold;
    border-radius: 20px;
    cursor: pointer;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease;
  }

  .card:hover { transform: scale(1.05); }

  .preview-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 10px;
  }

  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    color: white;
    text-align: center;
  }

  .plus-icon {
    font-size: 80px;
    margin-bottom: 20px;
    opacity: 0.9;
  }

  .placeholder p {
    font-size: 20px;
    margin: 0;
    opacity: 0.8;
  }

  .card::before,
  .card::after {
    position: absolute;
    content: "";
    width: 20%;
    height: 20%;
    background-color: rgba(255, 105, 255, 0.4);
    transition: all 0.6s ease;
    border-radius: 50%;
  }

  .card::before { top: 0; right: 0; border-radius: 0 20px 0 100%; }
  .card::after { bottom: 0; left: 0; border-radius: 0 100% 0 20px; }

  .card:hover::before,
  .card:hover::after {
    width: 100%;
    height: 100%;
    border-radius: 20px;
    background-color: rgba(82, 39, 255, 0.3);
  }
`;

export default UploadCard;
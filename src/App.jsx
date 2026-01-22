import React, { useState } from 'react';
import LightPillar from './LightPillar.jsx';
import TextType from './TextType.jsx';
import UploadCard from './UploadCard.jsx';
import ClassifyButton from './ClassifyButton.jsx';
import ResultsCards from './ResultsCard.jsx';
import ShinyText from './ShinyText.jsx';
import './App.css';

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [originalSize, setOriginalSize] = useState(null);

  const handleImageUpload = (file, dataUrl) => {
    setImageFile(file);
    setImagePreview(dataUrl);
    setResults(null);  // Reset results

    // Get original image size
    const img = new Image();
    img.onload = () => setOriginalSize(`${img.width} x ${img.height}`);
    img.src = dataUrl;
  };

  const handleClassify = async () => {
    if (!imageFile) {
      alert('Please upload an image first!');
      return;
    }

    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Prediction failed');

      const data = await response.json();
      setResults({
        bolt: data.bolt,
        locatingPin: data.locatingpin,
        nut: data.nut,
        washer: data.washer,
      });
    } catch (err) {
      console.error(err);
      alert('Error during classification. Is the backend running?');
    }

    setIsLoading(false);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden' }}>
      <LightPillar topColor="#5227FF" bottomColor="#FF9FFC" intensity={1.0} rotationSpeed={0.3} glowAmount={0.005} pillarWidth={3.0} pillarHeight={0.4} noiseIntensity={0.5} pillarRotation={0} interactive={false} mixBlendMode="normal" />

      {/* Header */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', padding: '40px 20px', textAlign: 'center', zIndex: 10, pointerEvents: 'none' }}>
        <TextType text={["Solid Works AI Hackathon", "Team : TouchGrass.exe"]} as="h1" typingSpeed={80} deletingSpeed={50} pauseDuration={3000} loop={true} showCursor={true} cursorCharacter="|" cursorBlinkDuration={0.6} className="typing-header"
          style={{ fontSize: '3.2rem', fontWeight: 'bold', color: '#ffffff', textShadow: '0 0 10px rgba(255, 105, 255, 0.8), 0 0 40px rgba(82, 39, 255, 0.6)', margin: 0, letterSpacing: '0.05em' }} />
      </div>

      {/* Upload Card */}
      <div style={{ position: 'absolute', top: '150px', left: '100px', zIndex: 5 }}>
        <UploadCard onImageUpload={handleImageUpload} />
      </div>

      {/* Classify Button - Only shown when NOT loading */}
      {!isLoading && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 15 }}>
          <ClassifyButton onClick={handleClassify} />
        </div>
      )}

      {/* Model Info + Original Size ONLY */}
      <div style={{ position: 'absolute', top: '62%', left: '50%', transform: 'translateX(-50%)', zIndex: 15, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px', fontFamily: 'monospace', textAlign: 'center' }}>
        {originalSize && <ShinyText text={`Input Size: ${originalSize}`} speed={1.8} color="#f8f8f8ff" shineColor="#8d8989ff" />}
      </div>

      {/* Results */}
      <div style={{ position: 'absolute', top: '350px', right: '120px', transform: 'translateY(-50%)', zIndex: 15 }}>
        <ResultsCards results={results} />
      </div>

      {/* Loading State - Styled like ClassifyButton */}
      {isLoading && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 20 }}>
          <div style={{
            fontSize: '1.5rem',
            padding: '0.2em 0.3em',
            letterSpacing: '0.1em',
            fontFamily: 'Orbitron, Courier New, monospace',
            fontWeight: '800',
            borderRadius: '0.8em',
            overflow: 'hidden',
            border: '3px solid #FF9FFC',
            background: 'linear-gradient(to right, rgba(82, 39, 255, 0.15) 1%, transparent 40%, transparent 60%, rgba(255, 159, 252, 0.15) 100%)',
            color: '#FF9FFC',
            textShadow: '0 0 15px rgba(255, 159, 252, 0.8)',
            boxShadow: 'inset 0 0 20px rgba(82, 39, 255, 0.3), 0 0 20px 5px rgba(255, 159, 252, 0.2)',
            pointerEvents: 'none',
          }}>
            CLASSIFYING...
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
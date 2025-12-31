// script.js
(() => {
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('imageUpload');
  const placeholderText = document.getElementById('placeholderText');
  const imagePreview = document.getElementById('imagePreview');
  const predictBtn = document.getElementById('predictBtn');
  const diagnosisResult = document.getElementById('diagnosisResult');
  const diseaseName = document.getElementById('diseaseName');
  const confidenceValue = document.getElementById('confidenceValue');
  const resultBox = document.getElementById('resultBox');
  const placeholderResultText = resultBox.querySelector('.placeholder-text');

  const PREDICT_URL = 'http://localhost:8000/predict';

  let selectedFile = null;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
      dropArea.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
      dropArea.classList.remove('drag-over');
    });
  });

  // When user drops file
  dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
  });

  // Click to open file dialog
  dropArea.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

  function handleFiles(files) {
    if (!files || files.length === 0) return;
    const file = files[0];

    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (jpg, png, etc.).');
      return;
    }

    selectedFile = file;
    showPreview(file);
    enablePredict(true);
  }

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      placeholderText.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  function enablePredict(enable) {
    predictBtn.disabled = !enable;
    if (enable) predictBtn.style.opacity = '1';
    else predictBtn.style.opacity = '0.6';
  }

  // Reset result UI
  function clearResult() {
    diagnosisResult.style.display = 'none';
    placeholderResultText.style.display = 'block';
  }

  // Handle predict click
  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      alert('Please select an image first.');
      return;
    }

    // UI feedback
    predictBtn.disabled = true;
    const originalText = predictBtn.textContent;
    predictBtn.textContent = 'Predicting...';

    clearResult();

    try {
      const formData = new FormData();
      formData.append('file', selectedFile, selectedFile.name);

      const resp = await fetch(PREDICT_URL, {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${txt}`);
      }

      const data = await resp.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Display results
      placeholderResultText.style.display = 'none';
      diagnosisResult.style.display = 'block';
      diseaseName.textContent = data.prediction || 'Unknown';
      confidenceValue.textContent = (data.confidence !== undefined) ? `${(data.confidence * 100).toFixed(2)}%` : '---';

    } catch (err) {
      console.error('Prediction error:', err);
      alert('Prediction failed: ' + (err.message || 'Unknown error. Check backend console.'));
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = originalText;
    }
  });

  // initial state
  enablePredict(false);
  clearResult();
})();

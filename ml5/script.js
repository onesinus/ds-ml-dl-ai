const correctImages = [
    { src: "images/bear.jpg", label: "bear" },
    { src: "images/cat.png", label: "cat" },
    { src: "images/orange.jpg", label: "orange" }
  ];
  
  const incorrectImages = [
    { src: "images/apple.png", label: "apple", actual: "dog" },
    { src: "images/dadu.png", label: "dice", actual: "keyboard" },
    { src: "images/person.jpg", label: "people", actual: "horse" }
  ];
  
  let classifier;
  ml5.imageClassifier('MobileNet', () => {
    console.log("Model loaded");
    classifier = ml5.imageClassifier('MobileNet');
    loadExamples();
  });
  
  function loadExamples() {
    loadImageSet(correctImages, 'correct-classifications', true);
    loadImageSet(incorrectImages, 'incorrect-classifications', false);
  }
  
  function loadImageSet(images, containerId, isCorrect) {
    const container = document.getElementById(containerId);
    images.forEach((item, index) => {
      const img = new Image();
      img.src = item.src;
      img.onload = () => {
        const wrapper = document.createElement('div');
        wrapper.className = 'user-section';
        const imgCol = document.createElement('div');
        imgCol.className = 'image-column';
        imgCol.appendChild(img);
  
        const canvas = document.createElement('canvas');
        canvas.id = `${containerId}-chart-${index}`;
        const chartCol = document.createElement('div');
        chartCol.className = 'classification-column';
        chartCol.appendChild(canvas);
  
        wrapper.appendChild(imgCol);
        wrapper.appendChild(chartCol);
        container.appendChild(wrapper);

        
        classifier.classify(img, (results) => {
          const labels = results.map(r => r.label);
          const confidences = results.map(r => r.confidence * 100);
          createChart(canvas.id, labels, confidences, isCorrect ? item.label : item.actual);
        });
      };
    });
  }
  
  function createChart(canvasId, labels, data, expectedLabel = "") {
    const backgroundColors = labels.map(label =>
      label === expectedLabel ? "green" : "rgba(255, 99, 132, 0.5)"
    );
  
    new Chart(document.getElementById(canvasId), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Confidence (%)',
          data: data.map(d => d.toFixed(2)),
          backgroundColor: backgroundColors
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: true, max: 100 }
        }
      }
    });
  }
  
  // ==== User Image Upload Section ====
  const dropArea = document.getElementById('drop-area');
  const fileInput = document.getElementById('file-input');
  const imagePreview = document.getElementById('user-image');
  const classifyButton = document.getElementById('classify-button');
  const userOutput = document.getElementById('user-output');
  
  dropArea.addEventListener('click', () => fileInput.click());
  dropArea.addEventListener('dragover', e => e.preventDefault());
  dropArea.addEventListener('drop', e => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
  
  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return alert('Invalid file.');
    const reader = new FileReader();
    reader.onload = e => {
      imagePreview.src = e.target.result;
      userOutput.style.display = 'flex';
    };
    reader.readAsDataURL(file);
  }
  
  classifyButton.addEventListener('click', () => {
    classifier.classify(imagePreview, results => {
      if (results) {
        const labels = results.map(r => r.label);
        const confidences = results.map(r => r.confidence * 100);
        if(Chart.getChart("result-chart")) {
            Chart.getChart("result-chart")?.destroy()
        }          
        createChart("result-chart", labels, confidences);  
      }
    });
  });
  
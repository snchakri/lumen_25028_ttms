// Fake Timetable Data
const timetableData = {
  metadata: {
    institution: "Demo Engineering College",
    department: "Computer Science & Engineering",
    semester: "5th Semester",
    academicYear: "2024-25",
    section: "A",
    generatedOn: "2025-10-28 13:38:00",
    totalCredits: 18,
    totalCourses: 5
  },
  schedule: [
    {
      day: "Monday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS305",
          courseName: "Web Technologies Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS1",
          type: "lab",
          credits: 3
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS305",
          courseName: "Web Technologies Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS1",
          type: "lab",
          credits: 3
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        }
      ]
    },
    {
      day: "Tuesday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS302",
          courseName: "Database Management Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS302",
          courseName: "Database Management Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        }
      ]
    },
    {
      day: "Wednesday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS303",
          courseName: "Operating Systems Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS303",
          courseName: "Operating Systems Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        }
      ]
    },
    {
      day: "Thursday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS304",
          courseName: "Computer Networks Lab",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS304",
          courseName: "Computer Networks Lab",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        }
      ]
    },
    {
      day: "Friday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS301",
          courseName: "Data Structures Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS301",
          courseName: "Data Structures Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        }
      ]
    }
  ],
  facultyLoad: [
    {
      facultyId: "FAC101",
      facultyName: "Dr. Rajesh Kumar",
      totalHours: 22,
      courses: ["CS301", "CS303"]
    },
    {
      facultyId: "FAC102",
      facultyName: "Dr. Priya Sharma",
      totalHours: 20,
      courses: ["CS302", "CS305"]
    },
    {
      facultyId: "FAC103",
      facultyName: "Prof. Amit Singh",
      totalHours: 14,
      courses: ["CS304"]
    }
  ],
  roomUtilization: [
    { room: "A101", utilization: "85%", totalHours: 34 },
    { room: "A102", utilization: "78%", totalHours: 31 },
    { room: "LAB-CS1", utilization: "92%", totalHours: 37 },
    { room: "LAB-CS2", utilization: "88%", totalHours: 35 },
    { room: "B201", utilization: "75%", totalHours: 30 }
  ]
};

// Application State
let currentData = timetableData;
let currentView = 'grid';
let selectedDay = 'all';
let searchQuery = '';

// DOM Elements
const modal = document.getElementById('timetableModal');
const viewTimetableBtn = document.getElementById('viewTimetableBtn');
const uploadCSVBtn = document.getElementById('uploadCSVBtn');
const closeModalBtn = document.getElementById('closeModalBtn');
const modalOverlay = document.querySelector('.modal-overlay');
const tabBtns = document.querySelectorAll('.tab-btn');
const viewContents = document.querySelectorAll('.view-content');
const dayFilter = document.getElementById('dayFilter');
const searchInput = document.getElementById('searchInput');
const exportBtn = document.getElementById('exportBtn');
const exportMenu = document.getElementById('exportMenu');
const darkModeToggle = document.getElementById('darkModeToggle');
const csvFileInput = document.getElementById('csvFileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadZone = document.getElementById('uploadZone');

// Initialize App
function init() {
  renderMetadata();
  renderGridView();
  renderStatistics();
  attachEventListeners();
}

// Render Metadata
function renderMetadata() {
  document.getElementById('institutionName').textContent = currentData.metadata.institution;
  document.getElementById('departmentName').textContent = currentData.metadata.department;
  document.getElementById('semesterName').textContent = currentData.metadata.semester;
}

// Render Grid View
function renderGridView() {
  const gridContainer = document.getElementById('timetableGrid');
  gridContainer.innerHTML = '';
  
  const days = selectedDay === 'all' 
    ? currentData.schedule.map(d => d.day) 
    : [selectedDay];
  
  const filteredSchedule = selectedDay === 'all' 
    ? currentData.schedule 
    : currentData.schedule.filter(d => d.day === selectedDay);
  
  // Get all unique time slots
  const allClasses = filteredSchedule.flatMap(d => d.classes);
  const timeSlots = [...new Set(allClasses.map(c => c.time))];
  
  // Create header row
  const cornerCell = document.createElement('div');
  cornerCell.className = 'grid-header';
  cornerCell.textContent = 'Time';
  gridContainer.appendChild(cornerCell);
  
  days.forEach(day => {
    const dayCell = document.createElement('div');
    dayCell.className = 'grid-header';
    dayCell.textContent = day;
    gridContainer.appendChild(dayCell);
  });
  
  // Create time rows
  timeSlots.forEach(time => {
    // Time cell
    const timeCell = document.createElement('div');
    timeCell.className = 'grid-cell time-cell';
    timeCell.textContent = time;
    gridContainer.appendChild(timeCell);
    
    // Class cells for each day
    filteredSchedule.forEach(daySchedule => {
      const classAtTime = daySchedule.classes.find(c => c.time === time);
      const classCell = document.createElement('div');
      classCell.className = 'grid-cell';
      
      if (classAtTime) {
        if (classAtTime.type === 'break') {
          classCell.classList.add('class-cell', 'break');
          classCell.innerHTML = `
            <div class="class-name">${classAtTime.courseName}</div>
          `;
        } else {
          classCell.classList.add('class-cell', classAtTime.type);
          classCell.innerHTML = `
            <div class="class-code">${classAtTime.courseCode}</div>
            <div class="class-name">${classAtTime.courseName}</div>
            <div class="class-faculty">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              ${classAtTime.facultyName}
            </div>
            <div class="class-room">${classAtTime.room}</div>
          `;
          
          // Add hover tooltip
          classCell.addEventListener('mouseenter', (e) => showTooltip(e, classAtTime, daySchedule.day));
          classCell.addEventListener('mouseleave', hideTooltip);
        }
      }
      
      gridContainer.appendChild(classCell);
    });
  });
  
  // Adjust grid columns
  gridContainer.style.gridTemplateColumns = `100px repeat(${days.length}, 1fr)`;
}

// Render List View
function renderListView() {
  const listContainer = document.getElementById('listContainer');
  listContainer.innerHTML = '';
  
  const filteredSchedule = selectedDay === 'all' 
    ? currentData.schedule 
    : currentData.schedule.filter(d => d.day === selectedDay);
  
  filteredSchedule.forEach(daySchedule => {
    const daySection = document.createElement('div');
    daySection.className = 'day-section';
    
    const dayHeader = document.createElement('div');
    dayHeader.className = 'day-header';
    dayHeader.textContent = daySchedule.day;
    daySection.appendChild(dayHeader);
    
    const classList = document.createElement('div');
    classList.className = 'class-list';
    
    const filteredClasses = daySchedule.classes.filter(c => 
      c.type !== 'break' && 
      (searchQuery === '' || 
       c.courseName.toLowerCase().includes(searchQuery.toLowerCase()) ||
       c.courseCode?.toLowerCase().includes(searchQuery.toLowerCase()) ||
       c.facultyName?.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    
    filteredClasses.forEach(classItem => {
      const item = document.createElement('div');
      item.className = 'class-item';
      item.innerHTML = `
        <div class="time-badge">${classItem.time}</div>
        <div class="class-details">
          <h4>${classItem.courseCode} - ${classItem.courseName}</h4>
          <div class="class-meta">
            <span>👤 ${classItem.facultyName}</span>
            <span>📍 ${classItem.room}</span>
            <span>📚 ${classItem.credits} Credits</span>
          </div>
        </div>
        <span class="type-badge ${classItem.type}">${classItem.type.toUpperCase()}</span>
      `;
      classList.appendChild(item);
    });
    
    daySection.appendChild(classList);
    listContainer.appendChild(daySection);
  });
}

// Render Faculty View
function renderFacultyView() {
  const facultyContainer = document.getElementById('facultyContainer');
  facultyContainer.innerHTML = '';
  
  currentData.facultyLoad.forEach(faculty => {
    const facultyCard = document.createElement('div');
    facultyCard.className = 'faculty-card';
    
    const facultyHeader = document.createElement('div');
    facultyHeader.className = 'faculty-header';
    facultyHeader.innerHTML = `
      <h3>${faculty.facultyName}</h3>
      <div class="faculty-info">
        <span>📊 ${faculty.totalHours} hours/week</span>
        <span>📚 ${faculty.courses.length} courses</span>
      </div>
    `;
    facultyCard.appendChild(facultyHeader);
    
    const facultySchedule = document.createElement('div');
    facultySchedule.className = 'faculty-schedule';
    
    // Get all classes for this faculty
    const facultyClasses = [];
    currentData.schedule.forEach(day => {
      day.classes.forEach(cls => {
        if (cls.facultyId === faculty.facultyId && cls.type !== 'break') {
          facultyClasses.push({ ...cls, day: day.day });
        }
      });
    });
    
    facultyClasses.forEach(cls => {
      const scheduleItem = document.createElement('div');
      scheduleItem.className = `schedule-item ${cls.type}`;
      scheduleItem.innerHTML = `
        <div class="schedule-time">${cls.day} • ${cls.time}</div>
        <div class="schedule-course">${cls.courseCode} - ${cls.courseName}</div>
        <div class="schedule-meta">Room: ${cls.room} • ${cls.type.toUpperCase()}</div>
      `;
      facultySchedule.appendChild(scheduleItem);
    });
    
    facultyCard.appendChild(facultySchedule);
    facultyContainer.appendChild(facultyCard);
  });
}

// Render Room View
function renderRoomView() {
  const roomContainer = document.getElementById('roomContainer');
  roomContainer.innerHTML = '';
  
  currentData.roomUtilization.forEach(roomData => {
    const roomCard = document.createElement('div');
    roomCard.className = 'room-card';
    
    const roomHeader = document.createElement('div');
    roomHeader.className = 'room-header';
    roomHeader.innerHTML = `
      <h3>${roomData.room}</h3>
      <div class="room-info">
        <span>📊 ${roomData.utilization} utilized</span>
        <span>⏱️ ${roomData.totalHours} hours/week</span>
      </div>
    `;
    roomCard.appendChild(roomHeader);
    
    const roomSchedule = document.createElement('div');
    roomSchedule.className = 'room-schedule';
    
    // Get all classes for this room
    const roomClasses = [];
    currentData.schedule.forEach(day => {
      day.classes.forEach(cls => {
        if (cls.room === roomData.room && cls.type !== 'break') {
          roomClasses.push({ ...cls, day: day.day });
        }
      });
    });
    
    roomClasses.forEach(cls => {
      const scheduleItem = document.createElement('div');
      scheduleItem.className = `schedule-item ${cls.type}`;
      scheduleItem.innerHTML = `
        <div class="schedule-time">${cls.day} • ${cls.time}</div>
        <div class="schedule-course">${cls.courseCode} - ${cls.courseName}</div>
        <div class="schedule-meta">Faculty: ${cls.facultyName} • ${cls.type.toUpperCase()}</div>
      `;
      roomSchedule.appendChild(scheduleItem);
    });
    
    roomCard.appendChild(roomSchedule);
    roomContainer.appendChild(roomCard);
  });
}

// Render Statistics
function renderStatistics() {
  let totalClasses = 0;
  let totalHours = 0;
  
  currentData.schedule.forEach(day => {
    day.classes.forEach(cls => {
      if (cls.type !== 'break') {
        totalClasses++;
        totalHours++;
      }
    });
  });
  
  document.getElementById('totalClasses').textContent = totalClasses;
  document.getElementById('totalHours').textContent = totalHours;
  document.getElementById('totalCredits').textContent = currentData.metadata.totalCredits;
}

// Show Tooltip
function showTooltip(event, classData, day) {
  const tooltip = document.getElementById('detailTooltip');
  const content = tooltip.querySelector('.tooltip-content');
  
  content.innerHTML = `
    <h4>${classData.courseCode} - ${classData.courseName}</h4>
    <p><strong>Day:</strong> ${day}</p>
    <p><strong>Time:</strong> ${classData.time}</p>
    <p><strong>Faculty:</strong> ${classData.facultyName}</p>
    <p><strong>Room:</strong> ${classData.room}</p>
    <p><strong>Type:</strong> ${classData.type.toUpperCase()}</p>
    <p><strong>Credits:</strong> ${classData.credits}</p>
  `;
  
  tooltip.style.display = 'block';
  tooltip.style.left = (event.pageX + 15) + 'px';
  tooltip.style.top = (event.pageY + 15) + 'px';
}

// Hide Tooltip
function hideTooltip() {
  const tooltip = document.getElementById('detailTooltip');
  tooltip.style.display = 'none';
}

// Switch View
function switchView(view) {
  currentView = view;
  
  // Update tabs
  tabBtns.forEach(btn => {
    if (btn.dataset.view === view) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  
  // Update views
  viewContents.forEach(content => {
    content.classList.remove('active');
  });
  
  const targetView = document.getElementById(`${view}View`);
  targetView.classList.add('active');
  
  // Render appropriate view
  switch(view) {
    case 'grid':
      renderGridView();
      break;
    case 'list':
      renderListView();
      break;
    case 'faculty':
      renderFacultyView();
      break;
    case 'room':
      renderRoomView();
      break;
  }
}

// Export Functions
function exportAsCSV() {
  let csv = 'Day,Time,Period,Course Code,Course Name,Faculty,Room,Type,Credits\n';
  
  currentData.schedule.forEach(day => {
    day.classes.forEach(cls => {
      if (cls.type !== 'break') {
        csv += `${day.day},${cls.time},${cls.period},${cls.courseCode},${cls.courseName},${cls.facultyName},${cls.room},${cls.type},${cls.credits}\n`;
      }
    });
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `timetable_${currentData.metadata.semester}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportAsHTML() {
  let html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Timetable - ${currentData.metadata.semester}</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #21808D; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #21808D; color: white; }
        .theory { background-color: rgba(59, 130, 246, 0.1); }
        .lab { background-color: rgba(34, 197, 94, 0.1); }
      </style>
    </head>
    <body>
      <h1>${currentData.metadata.institution}</h1>
      <h2>${currentData.metadata.department} - ${currentData.metadata.semester}</h2>
      <table>
        <thead>
          <tr>
            <th>Day</th>
            <th>Time</th>
            <th>Course Code</th>
            <th>Course Name</th>
            <th>Faculty</th>
            <th>Room</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  currentData.schedule.forEach(day => {
    day.classes.forEach(cls => {
      if (cls.type !== 'break') {
        html += `
          <tr class="${cls.type}">
            <td>${day.day}</td>
            <td>${cls.time}</td>
            <td>${cls.courseCode}</td>
            <td>${cls.courseName}</td>
            <td>${cls.facultyName}</td>
            <td>${cls.room}</td>
            <td>${cls.type.toUpperCase()}</td>
          </tr>
        `;
      }
    });
  });
  
  html += `
        </tbody>
      </table>
    </body>
    </html>
  `;
  
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `timetable_${currentData.metadata.semester}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

function printTimetable() {
  window.print();
}

// Parse CSV
function parseCSV(csvText) {
  const lines = csvText.split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  
  const schedule = {};
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    
    const values = lines[i].split(',').map(v => v.trim());
    const day = values[0];
    
    if (!schedule[day]) {
      schedule[day] = { day, classes: [] };
    }
    
    schedule[day].classes.push({
      time: values[1],
      period: parseInt(values[2]) || values[2],
      courseCode: values[3],
      courseName: values[4],
      facultyName: values[5],
      room: values[6],
      type: values[7],
      credits: parseInt(values[8]) || 0
    });
  }
  
  return Object.values(schedule);
}

// Handle CSV Upload
function handleCSVUpload(file) {
  const reader = new FileReader();
  
  reader.onload = (e) => {
    try {
      const csvText = e.target.result;
      const parsedSchedule = parseCSV(csvText);
      
      currentData.schedule = parsedSchedule;
      uploadZone.style.display = 'none';
      switchView('grid');
      renderStatistics();
    } catch (error) {
      alert('Error parsing CSV file. Please check the format.');
    }
  };
  
  reader.readAsText(file);
}

// Event Listeners
function attachEventListeners() {
  // Open/Close Modal
  viewTimetableBtn.addEventListener('click', () => {
    modal.classList.add('active');
  });
  
  uploadCSVBtn.addEventListener('click', () => {
    modal.classList.add('active');
    uploadZone.style.display = 'flex';
    viewContents.forEach(v => v.classList.remove('active'));
  });
  
  closeModalBtn.addEventListener('click', () => {
    modal.classList.remove('active');
    uploadZone.style.display = 'none';
    document.getElementById('gridView').classList.add('active');
  });
  
  modalOverlay.addEventListener('click', () => {
    modal.classList.remove('active');
    uploadZone.style.display = 'none';
    document.getElementById('gridView').classList.add('active');
  });
  
  // View Tabs
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      switchView(btn.dataset.view);
    });
  });
  
  // Day Filter
  dayFilter.addEventListener('change', (e) => {
    selectedDay = e.target.value;
    switchView(currentView);
  });
  
  // Search
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value;
    if (currentView === 'list') {
      renderListView();
    }
  });
  
  // Export
  exportBtn.addEventListener('click', () => {
    exportMenu.classList.toggle('active');
  });
  
  document.addEventListener('click', (e) => {
    if (!exportBtn.contains(e.target) && !exportMenu.contains(e.target)) {
      exportMenu.classList.remove('active');
    }
  });
  
  document.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', (e) => {
      const exportType = e.target.dataset.export;
      
      switch(exportType) {
        case 'csv':
          exportAsCSV();
          break;
        case 'html':
          exportAsHTML();
          break;
        case 'print':
          printTimetable();
          break;
      }
      
      exportMenu.classList.remove('active');
    });
  });
  
  // Dark Mode Toggle
  darkModeToggle.addEventListener('click', () => {
    document.documentElement.setAttribute(
      'data-theme',
      document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
    );
  });
  
  // CSV Upload
  browseBtn.addEventListener('click', () => {
    csvFileInput.click();
  });
  
  csvFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleCSVUpload(e.target.files[0]);
    }
  });
  
  uploadZone.addEventListener('click', () => {
    csvFileInput.click();
  });
  
  // Prevent modal close when clicking inside
  document.querySelector('.modal-container').addEventListener('click', (e) => {
    e.stopPropagation();
  });
}

// Initialize on load
init();
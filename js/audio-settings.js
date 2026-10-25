// Audio settings shared by the installation page (index.html) and the
// behind-the-scenes control panel (audio-control.html). The panel saves
// every change here and broadcasts it; the installation applies it live.
// Both pages must be opened from the same address in the same browser.
(function () {
  var STORAGE_KEY = "inSilicoAudio";
  var CHANNEL_NAME = "in-silico-audio";

  var DEFAULTS = {
    masterVolume: 1,
    speed: 0.75,
    keepPitch: true,
    pitchSemitones: 0,
    stemFirstVolume: 0.6,
    stemFullCount: 10,
    fadeInSeconds: 0.2,
    fadeOutSeconds: 2.5,
    entranceEnabled: true,
    entranceHoldSeconds: 1,
    entranceReleaseSeconds: 2.5,
    entranceDuck: 0.4,
    // Track 1 is the base layer; tracks 2-7 belong to the corners. Each
    // track's speed multiplies the overall speed, and its pitch (in
    // semitones) adds to the overall pitch.
    tracks: [
      { level: 0.5, mute: false, speed: 1, pitch: 0 },
      { level: 1, mute: false, speed: 1, pitch: 0 },
      { level: 1, mute: false, speed: 1, pitch: 0 },
      { level: 1, mute: false, speed: 1, pitch: 0 },
      { level: 1, mute: false, speed: 1, pitch: 0 },
      { level: 1, mute: false, speed: 1, pitch: 0 },
      { level: 1, mute: false, speed: 1, pitch: 0 },
    ],
  };

  var TRACKS = [
    { name: "Base", where: "always playing", gif: null },
    { name: "Top-left", where: "corner", gif: "bac1_1" },
    { name: "Top-right", where: "corner", gif: "bac2_1" },
    { name: "Bottom-left", where: "corner", gif: "bac3_1" },
    { name: "Bottom-right", where: "corner", gif: "bac4_1" },
    { name: "Left", where: "side", gif: "bac5_1" },
    { name: "Right", where: "side", gif: "bac1_2" },
  ];

  function copy(value) {
    return JSON.parse(JSON.stringify(value));
  }

  // Fills anything missing or malformed in a saved object from the defaults.
  function normalize(raw) {
    var settings = copy(DEFAULTS);
    if (!raw || typeof raw !== "object") return settings;
    Object.keys(DEFAULTS).forEach(function (key) {
      if (key === "tracks") return;
      var value = raw[key];
      if (typeof value === typeof DEFAULTS[key] && value === value) {
        settings[key] = value;
      }
    });
    if (Array.isArray(raw.tracks)) {
      settings.tracks.forEach(function (track, i) {
        var saved = raw.tracks[i];
        if (!saved) return;
        ["level", "speed", "pitch"].forEach(function (key) {
          if (typeof saved[key] === "number" && saved[key] === saved[key]) {
            track[key] = saved[key];
          }
        });
        if (typeof saved.mute === "boolean") track.mute = saved.mute;
      });
    }
    return settings;
  }

  function load() {
    try {
      return normalize(JSON.parse(localStorage.getItem(STORAGE_KEY)));
    } catch (err) {
      return copy(DEFAULTS);
    }
  }

  function save(settings) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (err) {}
  }

  // Some browsers refuse this on a page opened as a file; the page then
  // simply runs without the control panel.
  function openChannel() {
    if (typeof BroadcastChannel !== "function") return null;
    try {
      return new BroadcastChannel(CHANNEL_NAME);
    } catch (err) {
      return null;
    }
  }

  window.InSilicoAudio = {
    DEFAULTS: copy(DEFAULTS),
    TRACKS: TRACKS,
    defaults: function () {
      return copy(DEFAULTS);
    },
    normalize: normalize,
    load: load,
    save: save,
    openChannel: openChannel,
  };
})();

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
  Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
  Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
  Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
  Vec3& operator-=(const Vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
  Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
  float length() const { return sqrtf(x*x + y*y + z*z); }
  Vec3 normalized() const { float l=length(); return l>0.001f ? Vec3{x/l,y/l,z/l} : Vec3{0,0,1}; }
  float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
  Vec3 cross(const Vec3& o) const { return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x}; }
};

struct Quat {
  float w, x, y, z;
  Quat() : w(1), x(0), y(0), z(0) {}
  Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
  Quat operator*(const Quat& o) const {
    return {w*o.w - x*o.x - y*o.y - z*o.z,
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w};
  }
  Vec3 rotate(const Vec3& v) const {
    Quat p{0, v.x, v.y, v.z}, inv{w,-x,-y,-z}, r = *this * p * inv;
    return {r.x, r.y, r.z};
  }
  static Quat fromAxisAngle(const Vec3& axis, float angle) {
    float s = sinf(angle/2);
    return {cosf(angle/2), axis.x*s, axis.y*s, axis.z*s};
  }
  static Quat fromEuler(float yaw) {
    return fromAxisAngle({0,1,0}, yaw);
  }
  static Quat fromEuler(float yaw, float pitch, float roll) {
    Quat qy = fromAxisAngle({0,1,0}, yaw);
    Quat qp = fromAxisAngle({1,0,0}, pitch);
    Quat qr = fromAxisAngle({0,0,1}, roll);
    return qy * qp * qr;
  }
};

namespace Phys {
  constexpr float GRAVITY = -28.0f;
  constexpr float GROUND_FRICTION = 0.91f;
  constexpr float AIR_FRICTION = 0.97f;
  constexpr float BALL_FRICTION = 0.983f;
  constexpr float RESTITUTION = 0.6f;
  constexpr float WALL_RESTITUTION = 0.65f;
  constexpr float MAX_SPEED = 38.0f;
  constexpr float BOOST_FORCE = 750.0f;
  constexpr float ACCEL_FORCE = 550.0f;
  constexpr float BRAKE_FORCE = 350.0f;
  constexpr float TURN_SPEED = 3.5f;
  constexpr float LOW_SPEED_TURN = 2.0f;
  constexpr float JUMP_IMPULSE = 9.0f;
  constexpr float DODGE_IMPULSE = 16.0f;
  constexpr float BOOST_CONSUME = 8.0f;
  constexpr float BALL_RADIUS = 1.15f;
  constexpr float BALL_MASS = 1.0f;
  constexpr float CAR_MASS = 2.5f;
  constexpr float FW = 100.0f, FD = 82.0f, WH = 14.0f;
  constexpr float GW = 14.0f, GH = 8.0f, GD = 6.0f, WT = 0.5f;
}

struct CarInput {
  bool forward = false, backward = false, jump = false, boost = false;
  float steer = 0;
};

struct Car {
  Vec3 pos, vel;
  Quat quat;
  float yaw = 0, pitch = 0, roll = 0;
  float targetRoll = 0, targetPitch = 0;
  float boost = 100, dodgeTimer = 0, wheelSpin = 0;
  bool onGround = true, canJump = true, hasDoubleJumped = false, dodging = false, isBlue;
  int score = 0;

  void reset(Vec3 p) {
    pos = p; pos.y = 0.3f;
    vel = {0,0,0};
    yaw = isBlue ? 0 : float(M_PI);
    pitch = 0; roll = 0; targetRoll = 0; targetPitch = 0;
    quat = Quat::fromEuler(yaw); boost = 100;
    onGround = true; canJump = true; hasDoubleJumped = false; dodging = false;
  }
};

struct Ball {
  Vec3 pos, vel;
  float radius = Phys::BALL_RADIUS;
  float rotX = 0, rotZ = 0;
  void reset() { pos = {0,2,0}; vel = {0,0,0}; }
};

struct Particle {
  Vec3 pos, vel;
  float life, maxLife;
  float r, g, b;
};

std::vector<Particle> particles;
Car car1, car2;
Ball ball;
bool goalScored = false;
float gameTime = 300;
Uint32 lastTick = 0;
const Uint8* keys = nullptr;
float screenShake = 0;

const Vec3 boostPads[] = {
  {-25,0,16},{25,0,16},{-25,0,-16},{25,0,-16},
  {0,0,24},{0,0,-24},{-36,0,0},{36,0,0},
  {-15,0,28},{15,0,28},{-15,0,-28},{15,0,-28}
};
constexpr int numPads = sizeof(boostPads)/sizeof(boostPads[0]);

// Perspective projection (replaces gluPerspective)
static void myPerspective(float fovY, float aspect, float zNear, float zFar) {
  float f = 1.0f / tanf(fovY * float(M_PI) / 360.0f);
  float proj[16] = {
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (zFar+zNear)/(zNear-zFar), -1,
    0, 0, 2*zFar*zNear/(zNear-zFar), 0
  };
  glMultMatrixf(proj);
}

// LookAt matrix (replaces gluLookAt)
static void myLookAt(float ex, float ey, float ez,
                     float tx, float ty, float tz,
                     float ux, float uy, float uz) {
  Vec3 f{tx-ex, ty-ey, tz-ez}; f = f.normalized();
  Vec3 u{ux, uy, uz}; u = u.normalized();
  Vec3 s = f.cross(u).normalized();
  u = s.cross(f);
  float m[16] = {
     s.x,  u.x, -f.x, 0,
     s.y,  u.y, -f.y, 0,
     s.z,  u.z, -f.z, 0,
    -s.dot({ex,ey,ez}), -u.dot({ex,ey,ez}), f.dot({ex,ey,ez}), 1
  };
  glMultMatrixf(m);
}

void drawBox(float w, float h, float d, float r, float g, float b) {
  glColor3f(r,g,b);
  float hw=w/2, hh=h/2, hd=d/2;
  GLfloat v[] = {
    -hw,-hh,-hd,  hw,-hh,-hd,  hw, hh,-hd, -hw, hh,-hd,
    -hw,-hh, hd,  hw,-hh, hd,  hw, hh, hd, -hw, hh, hd
  };
  GLuint idx[] = {
    0,1,2, 2,3,0, 1,5,6, 6,2,1, 5,4,7, 7,6,5,
    4,0,3, 3,7,4, 3,2,6, 6,7,3, 4,5,1, 1,0,4
  };
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, v);
  glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, idx);
  glDisableClientState(GL_VERTEX_ARRAY);
}

void drawCylinder(float rad, float h, int seg, float r, float g, float b) {
  glColor3f(r,g,b);
  float hh = h/2;
  glBegin(GL_TRIANGLE_FAN);
  glVertex3f(0, -hh, 0);
  for (int i = 0; i <= seg; i++) {
    float a = 2*float(M_PI)*i/seg;
    glVertex3f(rad*cosf(a), -hh, rad*sinf(a));
  }
  glEnd();
  glBegin(GL_TRIANGLE_FAN);
  glVertex3f(0, hh, 0);
  for (int i = seg; i >= 0; i--) {
    float a = 2*float(M_PI)*i/seg;
    glVertex3f(rad*cosf(a), hh, rad*sinf(a));
  }
  glEnd();
  glBegin(GL_QUAD_STRIP);
  for (int i = 0; i <= seg; i++) {
    float a = 2*float(M_PI)*i/seg;
    float x=rad*cosf(a), z=rad*sinf(a);
    glVertex3f(x, -hh, z);
    glVertex3f(x, hh, z);
  }
  glEnd();
}

void drawSphere(float rad, int lat, int lon, float r, float g, float b) {
  glColor3f(r,g,b);
  for (int i = 0; i <= lat; i++) {
    float theta = float(M_PI)*i/lat;
    glBegin(GL_TRIANGLE_STRIP);
    for (int j = 0; j <= lon; j++) {
      float phi = 2*float(M_PI)*j/lon;
      float x = rad*sinf(theta)*cosf(phi);
      float y = rad*cosf(theta);
      float z = rad*sinf(theta)*sinf(phi);
      glVertex3f(x, y, z);
      float x2 = rad*sinf(theta+float(M_PI)/lat)*cosf(phi);
      float y2 = rad*cosf(theta+float(M_PI)/lat);
      float z2 = rad*sinf(theta+float(M_PI)/lat)*sinf(phi);
      glVertex3f(x2, y2, z2);
    }
    glEnd();
  }
}

void renderCar(const Car& c) {
  float r = c.isBlue ? 0.2f : 0.95f;
  float g = c.isBlue ? 0.5f : 0.55f;
  float b = c.isBlue ? 0.9f : 0.15f;
  float dark = 0.08f;
  float accent = c.isBlue ? 0.12f : 0.75f;
  float lightCol = c.isBlue ? 0.5f : 1.0f;

  glPushMatrix();
  glTranslatef(c.pos.x, c.pos.y, c.pos.z);

  float angle = 2 * acosf(c.quat.w) * 180 / float(M_PI);
  Vec3 axis{c.quat.x, c.quat.y, c.quat.z};
  if (axis.length() > 0.001f) {
    axis = axis.normalized();
    glRotatef(angle, axis.x, axis.y, axis.z);
  }

  // Main chassis body
  drawBox(1.8f, 0.4f, 3.2f, r, g, b);

  // Hood
  glPushMatrix(); glTranslatef(0, 0.1f, 1.5f);
  drawBox(1.5f, 0.1f, 1.0f, r*0.85f, g*0.85f, b*0.85f); glPopMatrix();

  // Front bumper
  glPushMatrix(); glTranslatef(0, 0.05f, 1.9f);
  drawBox(1.7f, 0.15f, 0.3f, r*0.6f, g*0.6f, b*0.6f); glPopMatrix();

  // Rear bumper
  glPushMatrix(); glTranslatef(0, 0.05f, -1.9f);
  drawBox(1.7f, 0.15f, 0.3f, r*0.6f, g*0.6f, b*0.6f); glPopMatrix();

  // Nose cone
  glPushMatrix(); glTranslatef(0, 0.25f, 1.95f);
  drawBox(1.2f, 0.25f, 0.35f, r*0.8f, g*0.8f, b*0.8f); glPopMatrix();

  // Rear diffuser
  glPushMatrix(); glTranslatef(0, 0.05f, -1.95f);
  drawBox(1.4f, 0.08f, 0.25f, dark, dark, dark); glPopMatrix();

  // Headlights
  glPushMatrix(); glTranslatef(-0.5f, 0.25f, 2.05f);
  drawBox(0.25f, 0.12f, 0.05f, 0.9f, 0.9f, 0.8f); glPopMatrix();
  glPushMatrix(); glTranslatef(0.5f, 0.25f, 2.05f);
  drawBox(0.25f, 0.12f, 0.05f, 0.9f, 0.9f, 0.8f); glPopMatrix();

  // Taillights
  glPushMatrix(); glTranslatef(-0.6f, 0.2f, -2.0f);
  drawBox(0.3f, 0.08f, 0.04f, 0.9f, 0.05f, 0.05f); glPopMatrix();
  glPushMatrix(); glTranslatef(0.6f, 0.2f, -2.0f);
  drawBox(0.3f, 0.08f, 0.04f, 0.9f, 0.05f, 0.05f); glPopMatrix();

  // Spoiler posts
  glPushMatrix(); glTranslatef(-0.5f, 0.65f, -1.7f);
  drawBox(0.06f, 0.3f, 0.06f, dark, dark, dark); glPopMatrix();
  glPushMatrix(); glTranslatef(0.5f, 0.65f, -1.7f);
  drawBox(0.06f, 0.3f, 0.06f, dark, dark, dark); glPopMatrix();

  // Spoiler wing
  glPushMatrix(); glTranslatef(0, 0.85f, -1.7f);
  drawBox(1.4f, 0.04f, 0.3f, accent, accent, accent); glPopMatrix();

  // Side skirts
  glPushMatrix(); glTranslatef(-0.95f, 0.1f, 0.5f);
  drawBox(0.06f, 0.12f, 1.8f, dark, dark, dark); glPopMatrix();
  glPushMatrix(); glTranslatef(0.95f, 0.1f, 0.5f);
  drawBox(0.06f, 0.12f, 1.8f, dark, dark, dark); glPopMatrix();

  // Windshield
  glPushMatrix(); glTranslatef(0, 0.5f, 1.2f);
  drawBox(1.2f, 0.3f, 0.08f, 0.25f, 0.45f, 0.85f); glPopMatrix();

  // Rear window
  glPushMatrix(); glTranslatef(0, 0.5f, -1.0f);
  drawBox(1.0f, 0.25f, 0.08f, 0.2f, 0.35f, 0.7f); glPopMatrix();

  // Exhaust pipes
  glPushMatrix(); glTranslatef(-0.25f, 0.05f, -2.0f);
  drawCylinder(0.06f, 0.12f, 8, dark, dark, dark); glPopMatrix();
  glPushMatrix(); glTranslatef(0.25f, 0.05f, -2.0f);
  drawCylinder(0.06f, 0.12f, 8, dark, dark, dark); glPopMatrix();
  glPushMatrix(); glTranslatef(-0.25f, 0.05f, -2.08f);
  drawCylinder(0.04f, 0.06f, 8, 0.3f, 0.3f, 0.3f); glPopMatrix();
  glPushMatrix(); glTranslatef(0.25f, 0.05f, -2.08f);
  drawCylinder(0.04f, 0.06f, 8, 0.3f, 0.3f, 0.3f); glPopMatrix();

  // Wheels with more detail
  float wp[4][3] = {{-0.75f,0.12f,1.1f},{0.75f,0.12f,1.1f},{-0.75f,0.12f,-1.2f},{0.75f,0.12f,-1.2f}};
  for (auto& w : wp) {
    glPushMatrix();
    glTranslatef(w[0], w[1], w[2]);
    glRotatef(c.wheelSpin, 1, 0, 0);
    // Tire
    drawCylinder(0.22f, 0.14f, 12, 0.04f, 0.04f, 0.04f);
    // Rim outer
    drawCylinder(0.14f, 0.15f, 8, accent, accent, accent);
    // Rim inner
    drawCylinder(0.06f, 0.16f, 6, dark, dark, dark);
    // Spokes
    glColor3f(accent, accent, accent);
    glBegin(GL_LINES);
    for (int s = 0; s < 5; s++) {
      float a = 2*float(M_PI)*s/5;
      glVertex3f(0.06f*cosf(a), 0, 0.06f*sinf(a));
      glVertex3f(0.13f*cosf(a), 0, 0.13f*sinf(a));
    }
    glEnd();
    // Tire tread lines
    glColor3f(0.06f, 0.06f, 0.06f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 16; i++) {
      float a = 2*float(M_PI)*i/16;
      glVertex3f(0.21f*cosf(a), 0.07f, 0.21f*sinf(a));
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 16; i++) {
      float a = 2*float(M_PI)*i/16;
      glVertex3f(0.21f*cosf(a), -0.07f, 0.21f*sinf(a));
    }
    glEnd();
    glPopMatrix();
  }

  glPopMatrix();
}

void renderBall(const Ball& b) {
  float r = b.radius;
  glPushMatrix();
  glTranslatef(b.pos.x, b.pos.y, b.pos.z);
  glRotatef(b.rotX*180/float(M_PI), 1, 0, 0);
  glRotatef(b.rotZ*180/float(M_PI), 0, 0, 1);

  // Main sphere - higher detail
  drawSphere(r, 18, 24, 0.92f, 0.92f, 0.92f);

  // Panel lines (soccer ball style seams)
  glColor3f(0.12f, 0.12f, 0.15f);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float a = 2*float(M_PI)*i/32;
    glVertex3f(r*0.97f*cosf(a), r*0.97f*sinf(a), 0);
  }
  glEnd();
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float a = 2*float(M_PI)*i/32;
    glVertex3f(0, r*0.97f*cosf(a), r*0.97f*sinf(a));
  }
  glEnd();
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float a = 2*float(M_PI)*i/32;
    glVertex3f(r*0.97f*cosf(a), 0, r*0.97f*sinf(a));
  }
  glEnd();

  // Equator band
  glColor3f(0.18f, 0.18f, 0.22f);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 36; i++) {
    float a = 2*float(M_PI)*i/36;
    glVertex3f(r*0.98f*cosf(a), r*0.02f, r*0.98f*sinf(a));
  }
  glEnd();
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 36; i++) {
    float a = 2*float(M_PI)*i/36;
    glVertex3f(r*0.98f*cosf(a), -r*0.02f, r*0.98f*sinf(a));
  }
  glEnd();

  // Subtle hexagon-like patches
  glColor3f(0.85f, 0.85f, 0.87f);
  for (int ring = 0; ring < 3; ring++) {
    float theta = float(M_PI) * (ring + 1) / 4;
    float y = r * cosf(theta);
    float rad = r * sinf(theta);
    int sides = 6 + ring * 2;
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0, y, 0);
    for (int i = 0; i <= sides; i++) {
      float a = 2*float(M_PI)*i/sides;
      glVertex3f(rad*cosf(a), y, rad*sinf(a));
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0, -y, 0);
    for (int i = 0; i <= sides; i++) {
      float a = 2*float(M_PI)*i/sides;
      glVertex3f(rad*cosf(a), -y, rad*sinf(a));
    }
    glEnd();
  }

  // Shiny highlight ring
  glColor3f(0.2f, 0.2f, 0.25f);
  glLineWidth(2);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float a = 2*float(M_PI)*i/32;
    glVertex3f(r*0.96f*cosf(a), r*0.35f*sinf(a), r*0.96f*sinf(a));
  }
  glEnd();
  glLineWidth(1);

  glPopMatrix();
}

void renderArena() {
  float hw = Phys::FW/2, hd = Phys::FD/2;

  glColor3f(0.12f, 0.25f, 0.12f);
  glBegin(GL_QUADS);
  glVertex3f(-hw, 0, -hd); glVertex3f(hw, 0, -hd);
  glVertex3f(hw, 0, hd); glVertex3f(-hw, 0, hd);
  glEnd();

  glColor3f(1,1,1);
  auto line = [](float x1,float z1,float x2,float z2) {
    glBegin(GL_LINES); glVertex3f(x1,0.01f,z1); glVertex3f(x2,0.01f,z2); glEnd();
  };
  line(-hw+1, -hd+1, hw-1, -hd+1);
  line(-hw+1, hd-1, hw-1, hd-1);
  line(-hw+1, -hd+1, -hw+1, hd-1);
  line(hw-1, -hd+1, hw-1, hd-1);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 36; i++) {
    float a = 2*float(M_PI)*i/36;
    glVertex3f(8*cosf(a), 0.01f, 8*sinf(a));
  }
  glEnd();
  glBegin(GL_LINES);
  glVertex3f(0, 0.01f, -8); glVertex3f(0, 0.01f, 8);
  glEnd();

  glColor3f(0.18f, 0.18f, 0.32f);
  auto wall = [](float x,float y,float z, float sx,float sz) {
    glPushMatrix(); glTranslatef(x,y,z);
    drawBox(sx, Phys::WH, sz, 0.18f, 0.18f, 0.32f);
    glPopMatrix();
  };
  wall(0, Phys::WH/2, -hd-0.25f, Phys::FW, 0.5f);
  wall(0, Phys::WH/2, hd+0.25f, Phys::FW, 0.5f);
  wall(-hw-0.25f, Phys::WH/2, 0, 0.5f, Phys::FD);
  wall(hw+0.25f, Phys::WH/2, 0, 0.5f, Phys::FD);

  auto goalPost = [](float x, float z) {
    glPushMatrix(); glTranslatef(x, Phys::GH/2, z);
    drawCylinder(0.12f, Phys::GH, 8, 0.85f, 0.85f, 0.85f);
    glPopMatrix();
  };
  for (int side = -1; side <= 1; side += 2) {
    float zz = side * (hd + 0.25f);
    goalPost(-Phys::GW/2, zz);
    goalPost(Phys::GW/2, zz);
    glPushMatrix(); glTranslatef(0, Phys::GH, zz);
    glRotatef(90, 1, 0, 0);
    drawCylinder(0.12f, Phys::GW, 8, 0.85f, 0.85f, 0.85f);
    glPopMatrix();
  }

  // Curved wall ramps (quarter-pipe at base of walls)
  glColor3f(0.25f, 0.25f, 0.35f);
  auto drawRamp = [](float cx, float cz, float len, int dirZ) {
    float rampRise = 3.0f, rampRun = 2.5f;
    int segs = 8;
    for (int i = 0; i < segs; i++) {
      float t0 = (float)i/segs;
      float t1 = (float)(i+1)/segs;
      float x0 = rampRun*(1-cosf(t0*float(M_PI)/2));
      float y0 = rampRise*sinf(t0*float(M_PI)/2);
      float x1 = rampRun*(1-cosf(t1*float(M_PI)/2));
      float y1 = rampRise*sinf(t1*float(M_PI)/2);
      float hlen = len/2;
      glBegin(GL_QUADS);
      // Front face
      glVertex3f(cx + x0, y0, cz + dirZ*0.05f);
      glVertex3f(cx + x1, y1, cz + dirZ*0.05f);
      glVertex3f(cx + x1, y1, cz + dirZ*0.05f + dirZ*hlen*2);
      glVertex3f(cx + x0, y0, cz + dirZ*0.05f + dirZ*hlen*2);
      // Top face
      glVertex3f(cx + x1, y1, cz + dirZ*0.05f);
      glVertex3f(cx + rampRun, rampRise, cz + dirZ*0.05f);
      glVertex3f(cx + rampRun, rampRise, cz + dirZ*0.05f + dirZ*hlen*2);
      glVertex3f(cx + x1, y1, cz + dirZ*0.05f + dirZ*hlen*2);
      glEnd();
    }
  };
  // Side wall ramps
  drawRamp(-hw+0.05f, 0, Phys::FD, 0); // left wall (facing +X)
  // We need to handle rotation - for now draw the four sides manually
  auto drawRampSide = [](float cx, float cz, float len, float ax, float az) {
    float rampRise = 3.0f, rampRun = 2.5f;
    int segs = 6;
    for (int i = 0; i < segs; i++) {
      float t0 = (float)i/segs;
      float t1 = (float)(i+1)/segs;
      float d0 = rampRun*(1-cosf(t0*float(M_PI)/2));
      float y0 = rampRise*sinf(t0*float(M_PI)/2);
      float d1 = rampRun*(1-cosf(t1*float(M_PI)/2));
      float y1 = rampRise*sinf(t1*float(M_PI)/2);
      float hlen = len/2;
      glBegin(GL_QUADS);
      glVertex3f(cx + ax*d0, y0, cz + az*d0);
      glVertex3f(cx + ax*d1, y1, cz + az*d1);
      glVertex3f(cx + ax*d1 + az*hlen, y1, cz + az*d1 - ax*hlen);
      glVertex3f(cx + ax*d0 + az*hlen, y0, cz + az*d0 - ax*hlen);
      glEnd();
    }
  };
  // Four wall ramps: -X, +X, -Z, +Z
  drawRampSide(-hw, 0, Phys::FD, 1, 0); // left wall
  drawRampSide(hw, 0, Phys::FD, -1, 0); // right wall
  drawRampSide(0, -hd, Phys::FW, 0, 1); // bottom wall (near)
  drawRampSide(0, hd, Phys::FW, 0, -1); // top wall (far)

  // Corner ramp sections
  auto drawCornerRamp = [](float cx, float cz, float dx, float dz) {
    float rampRise = 3.0f, rampRun = 2.5f;
    int segs = 6;
    for (int i = 0; i < segs; i++) {
      float t0 = (float)i/segs;
      float t1 = (float)(i+1)/segs;
      float d0 = rampRun*(1-cosf(t0*float(M_PI)/2));
      float y0 = rampRise*sinf(t0*float(M_PI)/2);
      float d1 = rampRun*(1-cosf(t1*float(M_PI)/2));
      float y1 = rampRise*sinf(t1*float(M_PI)/2);
      float s = rampRun;
      glBegin(GL_TRIANGLES);
      glVertex3f(cx, y0, cz);
      glVertex3f(cx + dx*d0, y0, cz + dz*d0);
      glVertex3f(cx + dx*d1, y1, cz + dz*d1);
      glEnd();
    }
  };
  drawCornerRamp(-hw+0.05f, -hd+0.05f, 1, 1);
  drawCornerRamp(-hw+0.05f, hd-0.05f, 1, -1);
  drawCornerRamp(hw-0.05f, -hd+0.05f, -1, 1);
  drawCornerRamp(hw-0.05f, hd-0.05f, -1, -1);

  // Boost pads
  glColor3f(0.3f, 0.6f, 1.0f);
  for (int i = 0; i < numPads; i++) {
    glPushMatrix();
    glTranslatef(boostPads[i].x, 0.02f, boostPads[i].z);
    glBegin(GL_QUADS);
    glVertex3f(-0.3f, 0, -0.3f); glVertex3f(0.3f, 0, -0.3f);
    glVertex3f(0.3f, 0, 0.3f); glVertex3f(-0.3f, 0, 0.3f);
    glEnd();
    glPopMatrix();
  }

  // Goal nets - white mesh grid
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  for (int side = -1; side <= 1; side += 2) {
    float zz = side * (hd + 0.4f);
    float netDepth = 3.0f;
    float startZ = zz;
    float endZ = zz + side * netDepth;
    float halfW = Phys::GW / 2 - 0.2f;

    glColor4f(1.0f, 1.0f, 1.0f, 0.35f);
    // Vertical net lines
    for (int i = 0; i <= 10; i++) {
      float x = -halfW + (2 * halfW * i / 10.0f);
      glBegin(GL_LINES);
      glVertex3f(x, 0.1f, startZ);
      glVertex3f(x, Phys::GH, startZ);
      glEnd();
      // Depth lines
      glBegin(GL_LINES);
      glVertex3f(x, 0.1f, startZ);
      glVertex3f(x, 0.1f, endZ);
      glEnd();
      glBegin(GL_LINES);
      glVertex3f(x, Phys::GH, startZ);
      glVertex3f(x, Phys::GH, endZ);
      glEnd();
    }
    // Horizontal net lines
    for (int i = 0; i <= 6; i++) {
      float y = 0.1f + (Phys::GH * i / 6.0f);
      glBegin(GL_LINES);
      glVertex3f(-halfW, y, startZ);
      glVertex3f(halfW, y, startZ);
      glEnd();
      glBegin(GL_LINES);
      glVertex3f(-halfW, y, endZ);
      glVertex3f(halfW, y, endZ);
      glEnd();
    }
    // Back net wall
    glColor4f(1.0f, 1.0f, 1.0f, 0.12f);
    glBegin(GL_QUADS);
    glVertex3f(-halfW, 0.1f, endZ);
    glVertex3f(halfW, 0.1f, endZ);
    glVertex3f(halfW, Phys::GH, endZ);
    glVertex3f(-halfW, Phys::GH, endZ);
    glEnd();
    // Back net grid
    glColor4f(1.0f, 1.0f, 1.0f, 0.25f);
    for (int i = 0; i <= 8; i++) {
      float x = -halfW + (2 * halfW * i / 8.0f);
      glBegin(GL_LINES);
      glVertex3f(x, 1, endZ);
      glVertex3f(x, Phys::GH-1, endZ);
      glEnd();
    }
    for (int i = 0; i <= 5; i++) {
      float y = 1 + ((Phys::GH-2) * i / 5.0f);
      glBegin(GL_LINES);
      glVertex3f(-halfW, y, endZ);
      glVertex3f(halfW, y, endZ);
      glEnd();
    }
  }

  // Glass panels (subtle)
  glColor4f(0.3f, 0.5f, 1.0f, 0.04f);
  for (int i = -3; i <= 3; i++) {
    glBegin(GL_QUADS);
    glVertex3f(i*10-5, 1, -hd-0.3f);
    glVertex3f(i*10+5, 1, -hd-0.3f);
    glVertex3f(i*10+5, Phys::WH-1, -hd-0.3f);
    glVertex3f(i*10-5, Phys::WH-1, -hd-0.3f);
    glEnd();
    glBegin(GL_QUADS);
    glVertex3f(i*10-5, 1, hd+0.3f);
    glVertex3f(i*10+5, 1, hd+0.3f);
    glVertex3f(i*10+5, Phys::WH-1, hd+0.3f);
    glVertex3f(i*10-5, Phys::WH-1, hd+0.3f);
    glEnd();
  }
  glDisable(GL_BLEND);
}

void spawnParticles(const Vec3& pos, const Vec3& vel, float r, float g, float b, int count) {
  for (int i = 0; i < count; i++) {
    Particle p;
    p.pos = pos;
    p.vel = vel + Vec3{float(rand()%100-50)/20, float(rand()%100-50)/20, float(rand()%100-50)/20};
    p.life = p.maxLife = 0.2f + (rand()%100)/400.0f;
    p.r = r; p.g = g; p.b = b;
    particles.push_back(p);
  }
}

void updateParticles(float dt) {
  for (int i = (int)particles.size()-1; i >= 0; i--) {
    particles[i].pos += particles[i].vel * dt;
    particles[i].life -= dt;
    if (particles[i].life <= 0) {
      particles[i] = particles.back();
      particles.pop_back();
    }
  }
}

void renderParticles() {
  if (particles.empty()) return;
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  glPointSize(5);
  glBegin(GL_POINTS);
  for (auto& p : particles) {
    float a = p.life / p.maxLife;
    glColor4f(p.r, p.g, p.b, a);
    glVertex3f(p.pos.x, p.pos.y, p.pos.z);
  }
  glEnd();
  glDisable(GL_BLEND);
}

void carBallCollision(Car& car, Ball& ball) {
  Vec3 d = ball.pos - car.pos;
  float dy = ball.pos.y - (car.pos.y + 0.4f);
  float dist = sqrtf(d.x*d.x + dy*dy + d.z*d.z);
  float minDist = Phys::BALL_RADIUS + 0.6f;

  if (dist < minDist && dist > 0.001f) {
    Vec3 n = d / dist;
    n.y = dy / dist;

    Vec3 rv = car.vel - ball.vel;
    float rvn = rv.dot(n);

    if (rvn > 0) {
      float imp = (1 + Phys::RESTITUTION) * rvn / (1/Phys::CAR_MASS + 1/Phys::BALL_MASS);
      ball.vel += n * (imp / Phys::BALL_MASS);
      ball.vel.y += fabsf(n.y) * imp / Phys::BALL_MASS * 0.8f;
      car.vel -= Vec3{n.x, 0, n.z} * (imp / Phys::CAR_MASS * 0.25f);
      screenShake = fminf(1, screenShake + 0.3f);
    }

    float overlap = (minDist - dist) * 0.5f;
    ball.pos += n * overlap * 0.6f;
    car.pos -= Vec3{n.x, 0, n.z} * overlap * 0.4f;
  }
}

void carCarCollision(Car& a, Car& b) {
  Vec3 d = b.pos - a.pos;
  float dist = sqrtf(d.x*d.x + d.z*d.z);
  if (dist < 2.0f && dist > 0.001f) {
    float overlap = (2.0f - dist) * 0.5f;
    Vec3 n = d / dist;
    a.pos -= n * overlap;
    b.pos += n * overlap;
    Vec3 rv = a.vel - b.vel;
    float rvn = rv.dot(n);
    if (rvn > 0) {
      // Bump strength: faster hit = harder bump
      float bumpFactor = 0.6f + 0.02f * fabsf(rvn);
      a.vel -= n * (rvn * bumpFactor);
      b.vel += n * (rvn * bumpFactor);
      // Add slight upward pop to make it feel impactful
      if (rvn > 5) {
        a.vel.y += 1.0f;
        b.vel.y += 1.0f;
      }
      screenShake = fminf(1.0f, screenShake + 0.15f);
    }
  }
}

void resetPositions() {
  car1.reset({-8, 0.3f, 15});
  car2.reset({8, 0.3f, -15});
  particles.clear();
  ball.reset();
  goalScored = false;
}

void scoreGoal(bool isOrange) {
  if (goalScored) return;
  goalScored = true;
  if (isOrange) car2.score++; else car1.score++;
  screenShake = 2.0f;
  resetPositions();
}

void updateCar(Car& c, const CarInput& inp, float dt) {
  float hw = Phys::FW/2 - 0.5f, hd = Phys::FD/2 - 0.5f;

  bool wasOnGround = c.onGround;
  c.onGround = c.pos.y <= 0.05f;

  if (c.onGround && !wasOnGround) {
    c.vel.y = 0;
  }

  if (c.onGround) {
    c.hasDoubleJumped = false;
    c.canJump = true;
  }

  // Jump
  if (inp.jump && c.onGround && c.canJump) {
    c.vel.y = Phys::JUMP_IMPULSE;
    c.onGround = false;
    c.canJump = false;
  } else if (inp.jump && !c.onGround && !c.hasDoubleJumped && c.dodgeTimer <= 0) {
    c.vel.y = Phys::JUMP_IMPULSE * 0.8f;
    c.hasDoubleJumped = true;
    c.dodging = true;
    c.dodgeTimer = 0.25f;
    Vec3 fwd = c.quat.rotate({0,0,1});
    c.vel += fwd * Phys::DODGE_IMPULSE;
    c.targetPitch = -0.3f;
  }
  c.dodgeTimer -= dt;
  if (c.dodgeTimer <= 0) {
    c.dodging = false;
    c.dodgeTimer = 0;
  }

  // Steering
  float speed = c.vel.length();
  float turnFactor = fmaxf(0.15f, speed / Phys::MAX_SPEED);
  if (!c.dodging)
    c.yaw += inp.steer * Phys::TURN_SPEED * dt * turnFactor;

  // Car tilt (roll when turning)
  c.targetRoll = -inp.steer * 0.4f * fminf(1, speed / 10.0f);
  if (!c.onGround) c.targetRoll *= 0.3f;

  // Pitch when accelerating/braking
  if (c.onGround) {
    if (inp.forward) c.targetPitch = -0.08f;
    else if (inp.backward) c.targetPitch = 0.12f;
    else c.targetPitch = 0;
  }

  // Smoothly interpolate roll and pitch
  float lerp = 6 * dt;
  c.roll += (c.targetRoll - c.roll) * lerp;
  c.pitch += (c.targetPitch - c.pitch) * lerp;
  if (c.onGround) {
    c.roll *= 0.95f;
    c.pitch *= 0.95f;
  }

  c.quat = Quat::fromEuler(c.yaw, c.pitch, c.roll);

  // Acceleration
  if (c.onGround) {
    Vec3 fwd = c.quat.rotate({0,0,1});
    fwd.y = 0; fwd = fwd.normalized();
    float acc = 0;
    if (inp.forward) acc = Phys::ACCEL_FORCE;
    if (inp.backward) acc = -Phys::BRAKE_FORCE;
    c.vel += fwd * (acc * dt);
    c.vel.x *= Phys::GROUND_FRICTION;
    c.vel.z *= Phys::GROUND_FRICTION;
  } else {
    c.vel.x *= Phys::AIR_FRICTION;
    c.vel.z *= Phys::AIR_FRICTION;
  }

  // Boost
  if (inp.boost && c.boost > 0) {
    Vec3 fwd = c.quat.rotate({0,0,1});
    fwd.y = 0; fwd = fwd.normalized();
    c.vel += fwd * (Phys::BOOST_FORCE * dt);
    c.boost = fmaxf(0, c.boost - Phys::BOOST_CONSUME * dt);
    Vec3 back = c.quat.rotate({0,0,-1});
    spawnParticles(c.pos + Vec3{0,0.1f,0}, back*3 +
      Vec3{float(rand()%100-50)/30, float(rand()%100-50)/30, float(rand()%100-50)/30},
      c.isBlue?0.2f:0.95f, c.isBlue?0.5f:0.55f, c.isBlue?0.9f:0.15f, 3);
  }

  // Speed cap
  float hspeed = Vec3{c.vel.x, 0, c.vel.z}.length();
  if (hspeed > Phys::MAX_SPEED) {
    float r = Phys::MAX_SPEED / hspeed;
    c.vel.x *= r; c.vel.z *= r;
  }

  c.vel.y += Phys::GRAVITY * dt;
  c.pos += c.vel * dt;

  // Ground collision
  if (c.pos.y <= 0) {
    if (c.vel.y < -3) screenShake = fminf(1, screenShake + 0.2f);
    c.pos.y = 0;
    c.vel.y = 0;
    if (!c.onGround) {
      c.onGround = true;
      c.canJump = true;
    }
  }

  // Ramp collision for car
  {
    float rampRise = 3.0f, rampRun = 2.5f;
    float margin = 0.3f;
    auto rampPush = [&](float dist, float& posComp, float& velComp, float pushDir) {
      if (dist < rampRun + margin && c.pos.y < rampRise + margin) {
        float t = dist / (rampRun + margin);
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        float rampY = rampRise * sinf(t * float(M_PI) / 2);
        if (c.pos.y < rampY) {
          c.pos.y = rampY;
          float ny = cosf(t * float(M_PI) / 2);
          float nx = sinf(t * float(M_PI) / 2);
          if (pushDir < 0) nx = -nx;
          float vn = velComp * nx + c.vel.y * ny;
          if (vn < 0) {
            velComp -= nx * vn * 0.5f;
            c.vel.y -= ny * vn * 0.5f;
          }
          posComp = pushDir * (rampRun + margin - dist + rampRun * (1 - cosf(t * float(M_PI) / 2)));
          if (c.onGround) {
            c.vel.y = 1.0f;
            c.onGround = false;
          }
        }
      }
    };
    rampPush(c.pos.x + hw, c.pos.x, c.vel.x, 1.0f);
    rampPush(hw - c.pos.x, c.pos.x, c.vel.x, -1.0f);
    rampPush(c.pos.z + hd, c.pos.z, c.vel.z, 1.0f);
    rampPush(hd - c.pos.z, c.pos.z, c.vel.z, -1.0f);
  }

  // Wall collision
  float margin = 0.3f;
  if (c.pos.x > hw - margin) { c.pos.x = hw - margin; c.vel.x = -fabsf(c.vel.x) * Phys::WALL_RESTITUTION; }
  if (c.pos.x < -hw + margin) { c.pos.x = -hw + margin; c.vel.x = fabsf(c.vel.x) * Phys::WALL_RESTITUTION; }
  bool inGoal = fabsf(c.pos.x) < Phys::GW/2 && c.pos.y < Phys::GH;
  if (c.pos.z > hd - margin && !inGoal) { c.pos.z = hd - margin; c.vel.z = -fabsf(c.vel.z) * Phys::WALL_RESTITUTION; }
  if (c.pos.z < -hd + margin && !inGoal) { c.pos.z = -hd + margin; c.vel.z = fabsf(c.vel.z) * Phys::WALL_RESTITUTION; }
  if (inGoal) {
    if (c.pos.z > hd+Phys::WT) { c.pos.z = hd+Phys::WT; c.vel.z *= -0.5f; }
    if (c.pos.z < -hd-Phys::WT) { c.pos.z = -hd-Phys::WT; c.vel.z *= -0.5f; }
  }
  if (c.pos.y > Phys::WH) { c.pos.y = Phys::WH; c.vel.y *= -0.3f; }

  c.wheelSpin += speed * dt * 250;

  // Boost pads
  for (int i = 0; i < numPads; i++) {
    Vec3 d = c.pos - boostPads[i];
    if (d.x*d.x + d.z*d.z < 1.0f && c.boost < 95)
      c.boost = fminf(100, c.boost + 30);
  }
}

// Ramp collision for ball: push ball up along wall ramps
static void applyRampCollision(Ball& b) {
  float hw = Phys::FW/2, hd = Phys::FD/2;
  float rampRise = 3.0f, rampRun = 2.5f;
  float rad = b.radius;

  auto rampPush = [&](float dist, float& pos, float& vel, float pushDir) {
    if (dist < rampRun + rad && b.pos.y < rampRise + rad) {
      float t = dist / (rampRun + rad);
      if (t < 0) t = 0;
      if (t > 1) t = 1;
      float rampY = rampRise * sinf(t * float(M_PI) / 2);
      if (b.pos.y < rampY + rad) {
        b.pos.y = rampY + rad;
        float ny = cosf(t * float(M_PI) / 2);
        float nx = sinf(t * float(M_PI) / 2);
        if (pushDir < 0) nx = -nx;
        float vn = vel * nx + b.vel.y * ny;
        if (vn < 0) {
          vel -= nx * vn * (1 + Phys::WALL_RESTITUTION);
          b.vel.y -= ny * vn * (1 + Phys::WALL_RESTITUTION);
        }
        // Clamp to ramp surface
        pos = pushDir * (rampRun + rad - dist + rampRun * (1 - cosf(t * float(M_PI) / 2)));
      }
    }
  };

  // Check all four walls
  float dxN = b.pos.x + hw;
  rampPush(dxN, b.pos.x, b.vel.x, 1.0f);
  float dxP = hw - b.pos.x;
  rampPush(dxP, b.pos.x, b.vel.x, -1.0f);
  float dzN = b.pos.z + hd;
  rampPush(dzN, b.pos.z, b.vel.z, 1.0f);
  float dzP = hd - b.pos.z;
  rampPush(dzP, b.pos.z, b.vel.z, -1.0f);
}

void updateBall(Ball& b, float dt) {
  b.vel.y += Phys::GRAVITY * dt * 1.5f;
  b.pos += b.vel * dt;

  float hw = Phys::FW/2, hd = Phys::FD/2;

  if (b.pos.y < b.radius) {
    b.pos.y = b.radius;
    b.vel.y *= -Phys::RESTITUTION;
    b.vel.x *= Phys::BALL_FRICTION;
    b.vel.z *= Phys::BALL_FRICTION;
  }

  // Wall ramps (curved edges)
  applyRampCollision(b);

  // Flat wall bounds (after ramp, for fallback)
  if (b.pos.x > hw - b.radius) { b.pos.x = hw - b.radius; b.vel.x *= -Phys::WALL_RESTITUTION; }
  if (b.pos.x < -hw + b.radius) { b.pos.x = -hw + b.radius; b.vel.x *= -Phys::WALL_RESTITUTION; }

  bool inGoalZone = fabsf(b.pos.x) < Phys::GW/2 && b.pos.y < Phys::GH;
  if (!inGoalZone) {
    if (b.pos.z > hd - b.radius) { b.pos.z = hd - b.radius; b.vel.z *= -Phys::WALL_RESTITUTION; }
    if (b.pos.z < -hd + b.radius) { b.pos.z = -hd + b.radius; b.vel.z *= -Phys::WALL_RESTITUTION; }
  }

  if (inGoalZone) {
    if (b.pos.z > hd + Phys::GD && b.vel.z > 0) scoreGoal(true);
    if (b.pos.z < -hd - Phys::GD && b.vel.z < 0) scoreGoal(false);
  }

  b.rotX += b.vel.z * dt * 0.5f;
  b.rotZ -= b.vel.x * dt * 0.5f;
}

CarInput getAIInput(const Car& c, const Ball& b, float dt) {
  CarInput inp;
  Vec3 toBall = b.pos - c.pos;
  float dist = toBall.length();
  if (dist < 0.01f) return inp;
  toBall = toBall.normalized();
  Vec3 fwd = c.quat.rotate({0,0,1});

  // Determine which goal to attack (orange = -Z, blue = +Z)
  float targetZ = c.isBlue ? -1.0f : 1.0f;

  // If we have the ball or are close, head toward opponent goal
  bool hasBall = dist < 5.0f;
  Vec3 target;
  if (hasBall) {
    // Aim toward opponent goal
    target = Vec3{0, 0, targetZ} - c.pos;
    target = target.normalized();
  } else {
    // Chase the ball
    target = toBall;
  }

  float dot = fwd.x*target.x + fwd.z*target.z;
  float cross = fwd.x*target.z - fwd.z*target.x;

  inp.forward = dot > -0.1f;
  inp.backward = dot < -0.7f;
  inp.steer = fmaxf(-1, fminf(1, cross * 2.5f));

  // Aggressive boost usage
  bool shouldBoost = false;
  if (hasBall) shouldBoost = c.boost > 20;
  else if (dist > 15) shouldBoost = c.boost > 15;
  else if (dist < 8 && c.boost > 60) shouldBoost = true;

  // Don't boost if we need to turn sharply
  if (fabsf(cross) > 0.8f) shouldBoost = false;

  inp.boost = shouldBoost && c.boost > 5;

  // Jump over obstacles or to hit ball in air
  bool shouldJump = false;
  if (dist < 6 && b.pos.y > 1.5f && c.onGround) {
    shouldJump = true;
  }
  if (dist < 8 && b.pos.y > 3.0f && c.onGround) {
    shouldJump = true;
  }
  if (shouldJump && (rand()%100) < 30) inp.jump = true;

  // Recover if stuck on wall
  if (!c.onGround && c.dodgeTimer <= 0 && (rand()%100) < 2) {
    inp.jump = true;
  }

  // Avoid own goal (don't drive into it)
  float ownGoalZ = c.isBlue ? Phys::FD/2 : -Phys::FD/2;
  if (fabsf(c.pos.x) < Phys::GW/2 && fabsf(c.pos.z - ownGoalZ) < Phys::GD + 3) {
    inp.steer = (c.pos.x > 0) ? -1 : 1;
    inp.forward = false;
    inp.backward = true;
    inp.boost = true;
  }

  return inp;
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

  SDL_Window* win = SDL_CreateWindow("Rocket Derby", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                      1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
  SDL_GLContext ctx = SDL_GL_CreateContext(win);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_CULL_FACE);
  glClearColor(0.06f, 0.06f, 0.15f, 1);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

  GLfloat lightPos0[] = {60, 80, 40, 1};
  GLfloat lightAmb0[] = {0.25f, 0.25f, 0.35f, 1};
  GLfloat lightDiff0[] = {0.85f, 0.85f, 0.9f, 1};
  GLfloat lightSpec0[] = {0.3f, 0.3f, 0.3f, 1};
  glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
  glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb0);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff0);
  glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpec0);

  GLfloat lightAmb1[] = {0,0,0,1};
  GLfloat lightPos1[] = {-40, 30, -30, 1};
  GLfloat lightDiff1[] = {0.15f, 0.4f, 0.8f, 1};
  glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);
  glLightfv(GL_LIGHT1, GL_AMBIENT, lightAmb1);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, lightDiff1);

  glMaterialf(GL_FRONT, GL_SHININESS, 25);

  keys = SDL_GetKeyboardState(nullptr);
  car1.isBlue = true; car2.isBlue = false;
  car1.score = 0; car2.score = 0;
  resetPositions();

  // Camera state
  Vec3 camPos{0, 8, 18}, camTarget{0, 0, 0};
  bool running = true;
  lastTick = SDL_GetTicks();
  CarInput inp;

  while (running) {
    Uint32 now = SDL_GetTicks();
    float dt = (now - lastTick) / 1000.0f;
    if (dt > 0.033f) dt = 0.033f;
    if (dt < 0.001f) dt = 0.001f;
    lastTick = now;

    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) running = false;
      if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
      if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) {
        glViewport(0, 0, e.window.data1, e.window.data2);
      }
    }

    inp.forward = keys[SDL_SCANCODE_W];
    inp.backward = keys[SDL_SCANCODE_S];
    inp.boost = keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RCTRL];
    inp.jump = keys[SDL_SCANCODE_SPACE];
    inp.steer = 0;
    if (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT]) inp.steer = 1;
    if (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]) inp.steer = -1;

    if (!goalScored) {
      gameTime -= dt;
      if (gameTime < 0) gameTime = 0;
    }

    updateCar(car1, inp, dt);
    updateCar(car2, getAIInput(car2, ball, dt), dt);
    updateBall(ball, dt);
    carBallCollision(car1, ball);
    carBallCollision(car2, ball);
    carCarCollision(car1, car2);
    updateParticles(dt);

    // Screen shake decay
    screenShake *= 0.92f;
    if (screenShake < 0.01f) screenShake = 0;

    // Camera: smooth third-person follow
    float spd = car1.vel.length();
    Vec3 carFwd = car1.quat.rotate({0,0,1});
    carFwd.y = 0; carFwd = carFwd.normalized();
    Vec3 carUp{0,1,0};
    Vec3 carRight = carFwd.cross(carUp).normalized();

    float camDist = 7 + spd * 0.05f;
    float camHeight = 3 + spd * 0.015f;
    Vec3 idealTarget = car1.pos + carFwd * 2.0f;
    idealTarget.y = car1.pos.y + 0.5f;

    Vec3 idealPos = car1.pos +
      carFwd * (-camDist) +
      carUp * camHeight +
      carRight * (-car1.roll * 0.5f);
    idealPos.y = fmaxf(idealPos.y, car1.pos.y + 1.5f);

    float camLerp = 4.0f * dt;
    camPos += (idealPos - camPos) * camLerp;
    camTarget += (idealTarget - camTarget) * camLerp;

    // Apply screen shake
    Vec3 shake{0,0,0};
    if (screenShake > 0.01f) {
      float sx = screenShake * (float(rand()%100-50)/50.0f);
      float sy = screenShake * (float(rand()%100-50)/50.0f);
      shake = carRight * sx + carUp * sy * 0.5f;
    }
    Vec3 finalPos = camPos + shake;

    int vpW, vpH;
    SDL_GetWindowSize(win, &vpW, &vpH);
    glViewport(0, 0, vpW, vpH);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)vpW / vpH;
    myPerspective(70, aspect, 0.1f, 300);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    myLookAt(finalPos.x, finalPos.y, finalPos.z,
             camTarget.x, camTarget.y, camTarget.z,
             0, 1, 0);

    GLfloat lp[] = {60, 80, 40, 1};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    renderArena();
    renderCar(car1);
    renderCar(car2);
    renderBall(ball);
    renderParticles();

    // HUD
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, vpW, vpH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0,0,0,0.5f);
    float boxW = 200, boxH = 36;
    glBegin(GL_QUADS);
    glVertex2f(vpW/2-boxW/2, 8); glVertex2f(vpW/2+boxW/2, 8);
    glVertex2f(vpW/2+boxW/2, 8+boxH); glVertex2f(vpW/2-boxW/2, 8+boxH);
    glEnd();

    // Draw a digit using 7-segment-like quads at (cx,cy) with scale
    auto drawDigit = [](float cx, float cy, float s, int d, float r, float g, float b) {
      if (d < 0 || d > 9) return;
      bool seg[7]; // a b c d e f g (standard 7-segment)
      //      aaa
      //    f     b
      //    f     b
      //      ggg
      //    e     c
      //    e     c
      //      ddd
      for (int i = 0; i < 7; i++) seg[i] = false;
      switch (d) {
        case 0: seg[0]=1;seg[1]=1;seg[2]=1;seg[3]=1;seg[4]=1;seg[5]=1; break;
        case 1: seg[1]=1;seg[2]=1; break;
        case 2: seg[0]=1;seg[1]=1;seg[3]=1;seg[4]=1;seg[6]=1; break;
        case 3: seg[0]=1;seg[1]=1;seg[2]=1;seg[3]=1;seg[6]=1; break;
        case 4: seg[1]=1;seg[2]=1;seg[5]=1;seg[6]=1; break;
        case 5: seg[0]=1;seg[2]=1;seg[3]=1;seg[5]=1;seg[6]=1; break;
        case 6: seg[0]=1;seg[2]=1;seg[3]=1;seg[4]=1;seg[5]=1;seg[6]=1; break;
        case 7: seg[0]=1;seg[1]=1;seg[2]=1; break;
        case 8: seg[0]=1;seg[1]=1;seg[2]=1;seg[3]=1;seg[4]=1;seg[5]=1;seg[6]=1; break;
        case 9: seg[0]=1;seg[1]=1;seg[2]=1;seg[3]=1;seg[5]=1;seg[6]=1; break;
      }
      float w = s * 0.15f, h = s * 0.04f, gap = s * 0.02f;
      auto hseg = [&](float x, float y, float len) {
        glBegin(GL_QUADS);
        glVertex2f(x - len/2, y - h/2); glVertex2f(x + len/2, y - h/2);
        glVertex2f(x + len/2, y + h/2); glVertex2f(x - len/2, y + h/2);
        glEnd();
      };
      auto vseg = [&](float x, float y, float len) {
        glBegin(GL_QUADS);
        glVertex2f(x - h/2, y - len/2); glVertex2f(x + h/2, y - len/2);
        glVertex2f(x + h/2, y + len/2); glVertex2f(x - h/2, y + len/2);
        glEnd();
      };
      glColor3f(r,g,b);
      float half = s * 0.3f;
      if (seg[0]) hseg(cx, cy + half, s*0.5f);
      if (seg[1]) vseg(cx + half, cy + half*0.5f, half);
      if (seg[2]) vseg(cx + half, cy - half*0.5f, half);
      if (seg[3]) hseg(cx, cy - half, s*0.5f);
      if (seg[4]) vseg(cx - half, cy - half*0.5f, half);
      if (seg[5]) vseg(cx - half, cy + half*0.5f, half);
      if (seg[6]) hseg(cx, cy, s*0.5f);
    };

    // Score display: large centered numbers
    float scoreScale = 36.0f;
    float scoreY = 28.0f;
    float scoreCX = vpW/2.0f;

    // Blue score (left)
    drawDigit(scoreCX - 50, scoreY, scoreScale, car1.score, 0.2f, 0.5f, 0.9f);
    // Colon
    glColor3f(1,1,1);
    glBegin(GL_QUADS);
    glVertex2f(scoreCX - 5, scoreY + 6); glVertex2f(scoreCX + 5, scoreY + 6);
    glVertex2f(scoreCX + 5, scoreY + 14); glVertex2f(scoreCX - 5, scoreY + 14);
    glEnd();
    glBegin(GL_QUADS);
    glVertex2f(scoreCX - 5, scoreY - 14); glVertex2f(scoreCX + 5, scoreY - 14);
    glVertex2f(scoreCX + 5, scoreY - 6); glVertex2f(scoreCX - 5, scoreY - 6);
    glEnd();
    // Orange score (right)
    drawDigit(scoreCX + 50, scoreY, scoreScale, car2.score, 0.95f, 0.55f, 0.15f);

    // Timer display below score
    int mins = (int)gameTime / 60;
    int secs = (int)gameTime % 60;
    drawDigit(scoreCX - 18, scoreY + 48, 20, mins / 10, 1, 1, 1);
    drawDigit(scoreCX + 6, scoreY + 48, 20, mins % 10, 1, 1, 1);
    // Timer colon
    glBegin(GL_QUADS);
    glVertex2f(scoreCX + 22, scoreY + 50); glVertex2f(scoreCX + 28, scoreY + 50);
    glVertex2f(scoreCX + 28, scoreY + 55); glVertex2f(scoreCX + 22, scoreY + 55);
    glEnd();
    glBegin(GL_QUADS);
    glVertex2f(scoreCX + 22, scoreY + 42); glVertex2f(scoreCX + 28, scoreY + 42);
    glVertex2f(scoreCX + 28, scoreY + 47); glVertex2f(scoreCX + 22, scoreY + 47);
    glEnd();
    drawDigit(scoreCX + 40, scoreY + 48, 20, secs / 10, 1, 1, 1);
    drawDigit(scoreCX + 64, scoreY + 48, 20, secs % 10, 1, 1, 1);

    // Boost bar
    float bw = 120 * (car1.boost / 100.0f);
    glColor3f(0.2f, 0.6f, 1.0f);
    glBegin(GL_QUADS);
    glVertex2f(20, vpH-30); glVertex2f(20+bw, vpH-30);
    glVertex2f(20+bw, vpH-18); glVertex2f(20, vpH-18);
    glEnd();
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(20, vpH-30); glVertex2f(140, vpH-30);
    glVertex2f(140, vpH-18); glVertex2f(20, vpH-18);
    glEnd();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    SDL_GL_SwapWindow(win);
    SDL_Delay(1);
  }

  SDL_GL_DeleteContext(ctx);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}

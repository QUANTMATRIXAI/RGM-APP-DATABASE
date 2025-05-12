import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt
import datetime

# Initialize database connection and ORM model
engine = create_engine('sqlite:///shopping.db')
Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    price = Column(Float)
    description = Column(String(500))

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    password_hash = Column(String(60))  # bcrypt hash length is 60

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    quantity = Column(Integer)
    order_date = Column(DateTime, default=datetime.datetime.now)

# Create tables if not exist
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, password):
    if not username or not password:
        raise ValueError("Username and password are required.")
    
    existing_user = session.query(User).filter_by(username=username).first()
    if existing_user:
        raise ValueError("Username already exists.")
    
    new_user = User(
        username=username,
        password_hash=hash_password(password)
    )
    session.add(new_user)
    session.commit()

def login_user(username, password):
    user = session.query(User).filter_by(username=username).first()
    if not user:
        return False
    if not check_password(password, user.password_hash):
        return False
    # Set up session state for authentication
    st.session_state['logged_in'] = True
    st.session_state['username'] = username
    return True

def eda_dashboard():
    st.subheader("Analytics Dashboard")
    
    # Example: Total orders per day
    orders_by_day = session.query(Order.order_date).all()
    dates = [order[0].date() for order in orders_by_day]
    st.bar_chart(dates)

# Main application
def main():
    st.title("Shopping Website with EDA Analytics")

    tabs = st.tabs(["Home", "Products", "Cart", "Wishlist", "Analytics", "Register/Login"])

    with tabs[0]:  # Home tab
        st.write("Welcome to our shopping store! Explore our products and start shopping.")

    with tabs[1]:  # Products tab
        st.subheader("Our Products")
        # Example product listing (replace with actual data)
        products = session.query(Product).all()
        for product in products:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{product.name}**")
                st.write(product.description)
                st.write(f"Price: ${product.price}")
            with col2:
                if st.button("Add to Cart"):
                    # Add to cart logic
                    pass

    with tabs[2]:  # Cart tab
        if 'cart' not in st.session_state:
            st.session_state.cart = []
        st.subheader("Your Cart")
        for item in st.session_state.cart:
            st.write(f"{item['name']} - ${item['price']} x {item['quantity']}")
        # Add cart management UI here

    with tabs[3]:  # Wishlist tab
        if 'wishlist' not in st.session_state:
            st.session_state.wishlist = []
        st.subheader("Your Wishlist")
        for item in st.session_state.wishlist:
            st.write(f"{item['name']} - ${item['price']}")
        # Add wishlist management UI here

    with tabs[4]:  # Analytics tab
        if 'logged_in' not in st.session_state or not st.session_state.logged_in:
            st.error("Please log in to access analytics.")
        else:
            eda_dashboard()

    with tabs[5]:  # Register/Login tab
        col1, col2 = st.columns(2)
        
        with col1:  # Login form
            with st.form("login_form"):
                username_login = st.text_input("Username")
                password_login = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if login_user(username_login, password_login):
                        st.success("Logged in successfully!")
                    else:
                        st.error("Invalid credentials.")
        
        with col2:  # Registration form
            with st.form("register_form"):
                username_register = st.text_input("New Username")
                password_register = st.text_input("New Password", type="password")
                if st.form_submit_button("Register"):
                    try:
                        register_user(username_register, password_register)
                        st.success("Registration successful! Please login.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
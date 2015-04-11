class AccountController < ApplicationController
  # Be sure to include AuthenticationSystem in Application Controller instead
  include AuthenticatedSystem
  # If you want "remember me" functionality, add this before_filter to Application Controller
  before_filter :login_from_cookie
  before_filter :authorize, :only => ['signup']

  # say something nice, you goof!  something sweet.
  
  def index
      redirect_to :action => 'login'
  end

  def login
    return unless request.post?
    self.current_user = User.authenticate(params[:login], params[:password])
    if logged_in?
      if params[:remember_me] == "1"
        self.current_user.remember_me
        cookies[:auth_token] = { :value => self.current_user.remember_token , :expires => self.current_user.remember_token_expires_at }
      end
      redirect_back_or_default(:controller => '/study', :action => 'index')
    else
      flash[:error] = "Incorrect Username or Password"
      render :action => "login"
    end
  end

  def signup
    @user = User.new(params[:user])
    return unless request.post?
    @user.save!
    self.current_user = @user
    redirect_to :controller => '/study', :action => 'index'
    flash[:notice] = "Thanks for signing up!"
  rescue ActiveRecord::RecordInvalid
    render :action => 'signup'
  end
  
  def logout
    self.current_user.forget_me if logged_in?
    cookies.delete :auth_token
    reset_session
    flash[:notice] = "You have been logged out."
    redirect_back_or_default(:controller => '/account', :action => 'login')
  end
  def codelogout
    self.current_user.forget_me if logged_in?
    cookies.delete :auth_token
    reset_session
    session[:user] = params[:ses][:id]
    session[:exp] = params[:ses][:exp]
    redirect_to(params[:path])
  end
  def users
    @users = User.find(:all)
  end
  
  private
  
  def authorize
    if User.count > 0
      login_required
    else
      false
    end
  end
end
